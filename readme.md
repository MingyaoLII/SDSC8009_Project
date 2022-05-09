# Environment

    Python                        3.9.7
    ogb                           1.3.3
    pandas                        1.4.1
    networkx                      2.6.3
    matplotlib                    3.5.1
    torch                         1.11.0
    cuda-version                  11.3
    torch-cluster                 1.6.0
    torch-geometric               2.0.4
    torch-scatter                 2.0.9
    torch-sparse                  0.6.13
    torch-spline-conv             1.2.1

# Dataset

[ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) downloaded/saved for use in the directory `'./data'` by running [Code Architecture-Implementation-Dataset](#Code-Architecture).

# Experiments

1. Check the [environment](#Environment).

2. Run the `.ipynb` file to reproduce our experiment results.
   1. Run [Code Architecture-Implementation](#Code-Architecture) for initialization, data visualization, models and functions implementation.
   2. Run [Code Architecture-Results](#Code-Architecture) for results reproduction.
   3. Every method is coded independently and all the outputs are attached. Arguments in `args={...}`  can be edited in every method. You can jump to any part of interest following the [Code Architecture-Results](#Code-Architecture).

3. **Notice**:
   1. 16 GB GPU memory is preferred for reproducing results of GCN-based models, and at least 24 GB GPU memory for GAT-based models. Hardwares we used include NVIDIA Tesla P100-16GB and NVIDIA GeForce RTX 3090-24GB.
   2. There is an alarm that play sounds every time a model is finished training and evaluated. Please disable the code [Implementation-Functions-Alarm](#Code-Architecture) if you find it disturbing.

# Code Architecture

    .
    ├── Implementation
    │   ├── Import Pakages
    │   ├── Dataset
    │   ├── Data Visualization
    │   ├── Initialization
    │   ├── Logger
    │   ├── Models
    |   │   ├── GCN
    |   │   ├── GAT
    |   │   ├── Residual_GCN
    |   │   └── Residual_GAT
    │   └── Functions
    |       ├── Alarm
    |       ├── Data Preprocess
    |       ├── Node2Vec
    |       └── Main
    ├── Results
    │   ├── GCN
    |   │   ├── GCN
    |   │   ├── Residual_GCN
    |   │   ├── Residual_GCN+Node2Vec
    |   │   ├── Residual_GCN+Node2Vec+Label_Reuse
    |   │   ├── Residual_GCN+C&S
    |   │   └── Residual_GCN+Node2Vec+Label_Reuse+C&S
    |   └── GAT
    |       ├── GAT
    |       ├── Residual_GAT
    |       ├── Residual_GAT+Node2Vec
    |       ├── Residual_GAT+Node2Vec+Label_Reuse
    |       ├── Residual_GAT+C&S
    |       └── Residual_GAT+Node2Vec+Label_Reuse+C&S
    └── End

# Our Results

Model | Test Acc % (Mean ± Std) | Train Acc % (Mean ± Std)
:-|:-:|:-:
GCN|71.97 ± 0.48|74.91 ± 1.21
Residual GCN|72.61 ± 0.34|77.59 ± 0.55
Residual GCN + Node2Vec|72.37 ± 0.20|80.08 ± 0.54
Residual GCN + Node2Vec + Label Reuse|72.30 ± 0.39|90.42 ± 1.62
Residual GCN + Node2Vec + Label Reuse + C&S|72.22 ± 0.14|98.81 ± 0.02
Residual GCN + C&S|73.06 ± 0.19|98.23 ± 0.02
GAT|70.23 ± 0.17|73.34 ± 0.34
Residual GAT|72.08 ± 0.45|77.66 ± 0.47
Residual GAT + Node2Vec|71.96 ± 0.32|80.19 ± 0.44
Residual GAT + Node2Vec + Label Reuse|71.59 ± 0.29|93.47 ± 1.01
Residual GAT + Node2Vec + Label Reuse + C&S|72.26 ± 0.26|98.82 ± 0.01
Residual GAT + C&S|72.65 ± 0.22|98.24 ± 0.02
