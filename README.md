# FML-DGCN: Federated Multi-Label Learning based on Dynamic Graph Convolutional Network

Implementation of paper *"FML-DGCN: Federated Multi-Label Learning based on Dynamic Graph Convolutional Network"*

# Requirements

- python: 3.8.13
- torch: 1.10.0+cu113 
- torchnet: 1.10.0+cu113 
- torchvision: 1.10.0+cu113 
- fate: 1.8.0        

# Usage
The experimental environment is built based on the FATE framework. For the construction of FATE framework, please refer
to [Here](https://github.com/FederatedAI/FATE).

The functions of each directory are as follows:
- `experiments/`
  - `ablation_studies/`: Implementation of federated multi-label learning methods for **ablation** studies.
  - `main_experiments/`: Implementation of federated multi-label learning methods for **comparative** experiments.
- `federatedml/nn/backend/`
  - `gcn/`: Implementation of GCN-based multi-label learning methods.
  - `multi_label/`: Implementation of asymmetric loss and API for ResNet-101 backbone.
  - `pytorch/`: Implementation of the subclasses of `torch.utils.data.Dataset` for PASCAL VOC and MS-COCO datasets.
  - `utils/`: Implementation of model aggregation methods, dataset loading methods, logging, and evaluation metrics.
# References
*https://github.com/Yejin0111/ADD-GCN*
