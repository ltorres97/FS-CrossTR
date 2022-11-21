## Multi-Scale Cross-Attention Transformer via Graph Embeddings for Few-Shot Molecular Property Prediction

In this work, a few-shot GNN-Transformer architecture is proposed to face the challenge of low-data in molecular property prediction. It is demonstrated that FS-CrossTR achieves a superior performance on small data collections over standard graph-based methodologies.

A dual-branch Transformer encoder learns multi-scale representations of molecular graph embeddings across two separate branches for small and large patches. A cross-attention mechanism fuses information from both sides to learn complementary representations built on the exchangeable information of small and large branches.

This two-module network learns deep representations from graph-level embeddings. First, a GNN module encodes the structural information of molecular graphs as a set of node and edge features. Node embeddings are then converted into graph embedding representations. A multi-scale Transformer encoder exploits the contextual information of these vectorial embeddings to propagate deep representations across attention layers at different scales.

![ScreenShot](/plots/fs-crosstr.png?raw=true)

A meta-learning framework was explored to optimize model parameters across tasks and quickly adapt to new molecular properties on few-shot data. 

Extensive experiments on real multiproperty prediction data demonstrate the predictive power and stable performances of the proposed model when inferring specific target properties adaptively.

This repository provides the source code and datasets for the proposed work.

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Data Availability and Pre-Processing

The Tox21 and SIDER datasets are downloaded from [Data](http://snap.stanford.edu/gnn-pretrain/data/) (chem_dataset.zip). 

The script loader.py pre-processes the data and transforms SMILES strings into molecular graphs using RDKit.Chem. 

Data pre-processing and pre-trained models are implemented based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Code Usage

### Installation
We used the following Python packages for core development. We tested on Python 3.7.

```
- torch = 1.10.1
- torch-cluster = 1.5.9
- torch-geometric = 2.0.4
- torch-scatter = 2.0.9
- torch-sparse = 0.6.12
- torch-spline-conv = 1.2.1
- torchvision = 0.10.0
- vit-pytorch = 0.35.8
- scikit-learn = 1.0.2
- seaborn = 0.11.2
- scipy = 1.8.0
- numpy = 1.21.5
- tqdm = 4.50.0
- tensorflow = 2.8.0
- keras = 2.8.0
- tsnecuda = 3.0.1
- tqdm = 4.62.3
- matplotlib = 3.5.1
- pandas = 1.4.1
- networkx = 2.7.1
- rdkit
```

## References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.: Strategies for pre-training graph neural networks. CoRR abs/1905.12265 (2020). https://doi.org/10.48550/ARXIV.1905.12265

```
@inproceedings{
hu2020pretraining,
title={Strategies for Pre-training Graph Neural Networks},
author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJlWWJSFDH},
}

```

[2] Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: 34th International Conference on Machine Learning, ICML 2017, vol. 3 (2017). https://doi.org/10.48550/arXiv.1703.03400

```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}

```

[3] Chen, C. F., Fan, Q., & Panda, R. (2021). CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. In Proceedings of the IEEE International Conference on Computer Vision (pp. 347–356). Institute of Electrical and Electronics Engineers Inc. https://doi.org/10.1109/ICCV48922.2021.00041.

```
@inproceedings{Chen2021,
   doi = {10.1109/ICCV48922.2021.00041},
   isbn = {9781665428125},
   issn = {15505499},
   journal = {Proceedings of the IEEE International Conference on Computer Vision},
   pages = {347-356},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   title = {CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification},
   year = {2021},
}


```
[4] Vision Transformers with PyTorch. https://github.com/lucidrains/vit-pytorch

```
@misc{Phil Wang,
  author = {Phil Wang},
  title = {Vision Transformers},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lucidrains/vit-pytorch}},
}

```



