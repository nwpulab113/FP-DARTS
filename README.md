# FP-DARTS: Fast Parallel Differentiable Neural Architecture Search for Image Classification
##  Introduction

Neural Architecture Search (NAS) has made remarkable progress in automatic machine learning. However, it still suffers massive computing overheads limiting its wide applications. In this paper, we present an efficient search method, Fast Parallel Differential Neural Architecture Search (FP-DARTS). The proposed method is carefully designed from three levels to construct and train the super-network. Firstly, at the operation-level, to reduce the computational burden, different from the standard DARTS search space (8 operations), we decompose the operation set into two non-overlapping operator sub-sets (4 operations for each). Adopting these two reduced search spaces, two over-parameterized sub-networks are constructed. Secondly, at the channel-level, the partially-connected strategy is adopted, where each sub-network only adopts partial channels. Then these two sub-networks construct a two-parallel-path super-network by addition. Thirdly, at the training-level, the binary gate is introduced to control whether a path participates in the super-network training. It may suffer an unfair issue when using softmax to select the best input for intermediate nodes across two operator sub-sets. To tackle this problem, the sigmoid function is introduced, which measures the performance of operations without compression. Extensive experiments demonstrate the effectiveness of the proposed algorithm. Specifically, FP-DARTS achieves 2.50\% test error with only 0.08 GPU-days on CIFAR10, and a state-of-the-art top-1 error rate of 23.7\% on ImageNet using only 2.44 GPU-days for search.
More details can be found in our paper.
## Requirement
#### python
#### Pytorch
#### CUDA
## Usage
### Search on ImageNet
Use train_search_imagenet.py
### Retrain on Image
Use train_imagenet.py
## Citation
If you find this project useful in your research, please consider cite our paper.
