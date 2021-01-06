
## GateTD: Gated Tensor Decomposition for Knowledge Graph Completion

This codebase contains PyTorch implementation of the paper:

> GateTD: Gated Tensor Decomposition for Knowledge Graph Completion.
> Ting Jia, Xinyan Wang, Kuo Yang and Xuezhong Zhou.

### Link Prediction Results

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
WN18RR | 0.472 | 43.4 | 48.2 | 55.0
FB15k-237 | 0.347 | 25.3 | 38.0 | 53.8
WN18 | 0.951 | 94.6 | 95.5 | 96.0
FB15k | 0.854 | 82.7 | 86.9 | 90.4


### Running a model

To run the model, execute the following command:

     python learn.py --dataset FB15k-237 --max_epochs 1000 --batch_size 1024 --learning_rate 0.2
                                         --edim 2500 --rdim 1200 --reg 0.4 --init 0.001
                                     

Available datasets are:

    WN18RR
    FB15k-237
    WN18
    FB15k
    
To reproduce the results from the paper, use the following combinations of hyperparameters with `batch_size=128`:

dataset | batch_size | learning_rate | edim | rdim | reg | init
:--- | :---: | :---: | :---: | :---: | :---: | :---: 
WN18RR | 1024 | 0.3 | 1000 | 500 | 0.5 | 0.001
FB15k-237 | 1024 | 0.2 | 2500 | 1200 | 0.4 | 0.001 
WN18 | 1024 | 0.25 | 1000 | 1000 | 0.3 | 0.001 
FB15k | 1024 | 0.05 | 2500 | 1200 | 0.03 | 0.001 

    
### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1
