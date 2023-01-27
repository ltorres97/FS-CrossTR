import torch
from torch import nn, einsum
from torch_geometric.nn import global_mean_pool
from gnn_models import GNN
from torch_geometric.nn import MessagePassing
from einops import rearrange
from torch_geometric import utils
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from vit_pytorch.vit import Transformer


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Norm layer (Norm)

class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Feedforward network (FFN)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Attention-layer (MSA)

class MSA(nn.Module):
    def __init__(self, dim, heads = 5, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # CA needs the cls token to exchange information

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Large and Small Patch Transformer encoder

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Norm(dim, FFN(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# Function to adjust and project small and large patch embedding tokens

class ProjectFunction(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# Cross attention transformer (CA)

class CATransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectFunction(sm_dim, lg_dim, Norm(lg_dim, MSA(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectFunction(lg_dim, sm_dim, Norm(sm_dim, MSA(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# Multi-scale Transformer encoder

class MultiScaleTransformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CATransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# Convert 1D image tokens into patch embeddings - Adaptation for 1D visual fatures

class Sequence1DEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        emb_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        
        emb_height, emb_width = pair(emb_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (emb_height // patch_height) * (emb_width // patch_width)
        patch_dim = 1 * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# Cross-Attention Transformer adapted for 1D visual features - graph embeddings

class TR(nn.Module):
    def __init__(
        self,
       emb_size,
       num_classes,
       sm_dim,
       lg_dim,
       sm_patch_size = (30,1),
       sm_enc_depth = 3,
       sm_enc_heads = 5,
       sm_enc_mlp_dim = 1024,
       sm_enc_dim_head = 64,
       lg_patch_size = (60,1),
       lg_enc_depth = 3,
       lg_enc_heads = 5,
       lg_enc_mlp_dim = 1024,
       lg_enc_dim_head = 64,
       cross_attn_depth = 2,
       cross_attn_heads = 5,
       cross_attn_dim_head = 64,
       depth = 3,
       dropout = 0.1,
       emb_dropout = 0.1
    
    ):
        super().__init__()
        self.sm_image_embedder = Sequence1DEmbedder(dim = sm_dim, emb_size = emb_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = Sequence1DEmbedder(dim = lg_dim, emb_size = emb_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleTransformer(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, emb):
        
        emb = emb.reshape(10,1,300,1)
        
        small_tokens = self.sm_image_embedder(emb)
        large_tokens = self.lg_image_embedder(emb)

        small_tokens, large_tokens = self.multi_scale_encoder(small_tokens, large_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (small_tokens, large_tokens))

        sm_cls_logits = self.sm_mlp_head(sm_cls)
        lg_cls_logits = self.lg_mlp_head(lg_cls)

        return sm_cls_logits + lg_cls_logits #, sm_cls
    
    
class GNN_prediction(torch.nn.Module):

    def __init__(self, layer_number, emb_dim, jk = "last", dropout_prob= 0, pooling = "mean", gnn_type = "gin"):
        super(GNN_prediction, self).__init__()
        
        self.num_layer = layer_number
        self.drop_ratio = dropout_prob
        self.jk = jk
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of layers must be > 2.")

        self.gnn = GNN(layer_number, emb_dim, jk, dropout_prob, gnn_type = gnn_type)
        
        if pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Invalid pooling.")

        self.mult = 1
        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)
        
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:0'), strict = False)
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("The arguments are unmatched!")

        node_embeddings = self.gnn(x, edge_index, edge_attr)
            
        pred_gnn = self.graph_pred_linear(self.pool(node_embeddings, batch))
        
        return pred_gnn, node_embeddings
        
        
if __name__ == "__main__":
    pass
