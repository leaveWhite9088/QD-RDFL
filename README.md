# QD-RDFL

<div align="center">

# Quality-aware Dynamic Resources-decoupled Federated Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📋 Overview

In this paper, we focus on **data assetization**, where data does not circulate but the information does.

Our approach includes:

- A framework for **resource-decoupled FL** involving model owners, data owners, and computing centers
- A **Tripartite Stackelberg Model** with theoretical analysis of the Stackelberg-Nash equilibrium (SNE)
- The **Quality-aware Dynamic Resources-decoupled FL algorithm (QD-RDFL)**, which derives optimal strategies for all parties using backward induction
- A **dynamic optimization mechanism** that improves strategy profiles by evaluating data quality contributions

Our extensive experiments demonstrate that our method effectively encourages collaboration between the three parties involved, maximizing global utility and the value of data assets.

## 🖼️ Model Framework

![framework](data/imgs/framework.png)

## 🛠️ Requirements

```bash
torch==2.3.0
torchvision==0.18.1a0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
```

## 📂 Project Structure

```
.
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│  setup.py
│
├─data               # Data storage
│  ├─dataset         # MNIST, CIFAR10, CIFAR100 datasets
│  ├─imgs            # Result visualizations
│  ├─log             # Experiment logs
│  ├─model           # Trained models
│  └─saved           # Saved experimental results
│
└─src                # Source code
   └─qd_rdfl         # Main package
      │  __init__.py
      │  __main__.py
      │  global_variable.py
      │
      ├─algorithms   # Core algorithms
      │  │  __init__.py
      │  │  GaleShapley.py
      │  │  Stackelberg.py
      │
      ├─datasets     # Dataset loaders
      │  │  __init__.py  
      │  │  CIFAR100Dataset.py
      │  │  CIFAR10Dataset.py
      │  │  MNISTDataset.py
      │
      ├─experiments  # Experiment scripts
      │  │  __init__.py
      │  ├─ablation        # Ablation studies
      │  ├─comparison      # Method comparisons
      │  ├─parameter_analysis  # Parameter sensitivity analysis
      │  └─supplement      # Supplementary experiments
      │
      ├─models       # Model architectures
      │  │  __init__.py
      │  │  CIFAR100CNN.py
      │  │  CIFAR10CNN.py
      │  │  MNISTCNN.py
      │
      ├─plots        # Visualization scripts
      │  │  __init__.py
      │  ├─ablation
      │  ├─comparison
      │  ├─parameter_analysis
      │  └─supplement
      │
      ├─roles        # Participant role definitions
      │  │  __init__.py
      │  │  CPC.py         # Computing center
      │  │  DataOwner.py   # Data owner
      │  │  ModelOwner.py  # Model owner
      │
      └─utils        # Utility functions
         │  __init__.py
         │  UtilsCIFAR10.py
         │  UtilsCIFAR100.py
         │  UtilsMNIST.py
```

## 🚀 Running Experiments

Note: The `--parent_path` parameter specifies where logs will be stored. Optional values: log-parameter_analysis, log-comparison, log-ablation, log-supplement, log-main.

### ⚙️ Ablation Studies
Each experiment takes approximately 30 minutes due to training requirements.

#### With Dynamic Adjustment
```bash
# MNIST with dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.adjust.ablation-MNIST-adjust --adjustment_literation 2 --parent_path log-ablation

# CIFAR10 with dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.adjust.ablation-CIFAR10-adjust --adjustment_literation 2 --parent_path log-ablation

# CIFAR100 with dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.adjust.ablation-CIFAR100-adjust --adjustment_literation 2 --parent_path log-ablation
```

#### Without Dynamic Adjustment
```bash
# MNIST without dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.noneadjust.ablation-MNIST-noneadjust --adjustment_literation 2 --parent_path log-ablation

# CIFAR10 without dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.noneadjust.ablation-CIFAR10-noneadjust --adjustment_literation 2 --parent_path log-ablation

# CIFAR100 without dynamic adjustment
python -m src.qd_rdfl.experiments.ablation.noneadjust.ablation-CIFAR100-noneadjust --adjustment_literation 2 --parent_path log-ablation
```

### 📊 Comparison Experiments
Each comparison experiment takes about 30 minutes to run.

#### FIX Strategy
```bash
# MNIST with FIX strategy
python -m src.qd_rdfl.experiments.comparison.FIX.comparison-MNIST-FIX --adjustment_literation 2 --parent_path log-comparison

# CIFAR10 with FIX strategy
python -m src.qd_rdfl.experiments.comparison.FIX.comparison-CIFAR10-FIX --adjustment_literation 2 --parent_path log-comparison

# CIFAR100 with FIX strategy
python -m src.qd_rdfl.experiments.comparison.FIX.comparison-CIFAR100-FIX --adjustment_literation 2 --parent_path log-comparison
```

#### RANDOM Strategy
```bash
# MNIST with RANDOM strategy
python -m src.qd_rdfl.experiments.comparison.RANDOM.comparison-MNIST-RANDOM --adjustment_literation 2 --parent_path log-comparison

# CIFAR10 with RANDOM strategy
python -m src.qd_rdfl.experiments.comparison.RANDOM.comparison-CIFAR10-RANDOM --adjustment_literation 2 --parent_path log-comparison

# CIFAR100 with RANDOM strategy
python -m src.qd_rdfl.experiments.comparison.RANDOM.comparison-CIFAR100-RANDOM --adjustment_literation 2 --parent_path log-comparison
```

#### QD-RDFL (Our Method)
```bash
# MNIST with QD-RDFL
python -m src.qd_rdfl.experiments.comparison.QD-RDFL.comparison-MNIST-QDRDFL --adjustment_literation 2 --parent_path log-comparison

# CIFAR10 with QD-RDFL
python -m src.qd_rdfl.experiments.comparison.QD-RDFL.comparison-CIFAR10-QDRDFL --adjustment_literation 2 --parent_path log-comparison

# CIFAR100 with QD-RDFL
python -m src.qd_rdfl.experiments.comparison.QD-RDFL.comparison-CIFAR100-QDRDFL --adjustment_literation 2 --parent_path log-comparison
```

#### AGGR-NIID (Three ways to run together)
```bash
# MNIST with AGGR-NIID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-NIID.comparison-MNIST-AGGRNIID --adjustment_literation 2 --parent_path log-comparison

# CIFAR10 with AGGR-NIID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-NIID.comparison-CIFAR10-AGGRNIID --adjustment_literation 2 --parent_path log-comparison

# CIFAR100 with AGGR-NIID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-NIID.comparison-CIFAR100-AGGRNIID --adjustment_literation 2 --parent_path log-comparison

# IMDB with AGGR-NIID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-NIID.comparison-IMDB-AGGRNIID --adjustment_literation 2 --parent_path log-comparison
```

#### AGGR-IID (Three ways to run together)
```bash
# MNIST with AGGR-IID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-IID.comparison-MNIST-AGGRIID --adjustment_literation 2 --parent_path log-comparison

# CIFAR10 with AGGR-IID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-IID.comparison-CIFAR10-AGGRIID --adjustment_literation 2 --parent_path log-comparison

# CIFAR100 with AGGR-IID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-IID.comparison-CIFAR100-AGGRIID --adjustment_literation 2 --parent_path log-comparison

# IMDB with AGGR-IID strategy
python -m src.qd_rdfl.experiments.comparison.AGGR-IID.comparison-IMDB-AGGRIID --adjustment_literation 2 --parent_path log-comparison
```

### 📈 Parameter Analysis
Parameter analysis experiments run quickly (approximately 30 seconds each).

#### Eta Parameter
Investigate changes in Us under different Eta values:
```bash
python -m src.qd_rdfl.experiments.parameter_analysis.Eta.parameter_analysis-MNIST-Eta --adjustment_literation -1 --parent_path log-parameter_analysis
```

#### L Parameter
Verify the impact of adjustment rounds:
```bash
python -m src.qd_rdfl.experiments.parameter_analysis.L.parameter_analysis-MNIST-L --adjustment_literation -1 --parent_path log-parameter_analysis
```

#### xn Parameter
Investigate changes in average Un under different xn values:
```bash
python -m src.qd_rdfl.experiments.parameter_analysis.xn.parameter_analysis-MNIST-xn --adjustment_literation -1 --parent_path log-parameter_analysis
```

### 📋 Supplementary Experiments

#### Accuracy Analysis
```bash
python -m src.qd_rdfl.experiments.supplement.accurancy.supplement-MNIST-accurancy --adjustment_literation -1 --parent_path log-supplement

python -m src.qd_rdfl.experiments.supplement.accurancy.supplement-CIFAR10-accurancy --adjustment_literation -1 --parent_path log-supplement
```

#### Un (Data Owner Utility) Analysis
```bash
python -m src.qd_rdfl.experiments.supplement.Un.supplement-MNIST-Un --adjustment_literation -1 --parent_path log-supplement

python -m src.qd_rdfl.experiments.supplement.Un.supplement-CIFAR10-Un --adjustment_literation -1 --parent_path log-supplement
```

#### Us (Model Owner Utility) Analysis
```bash
python -m src.qd_rdfl.experiments.supplement.Us.supplement-MNIST-Us --adjustment_literation -1 --parent_path log-supplement

python -m src.qd_rdfl.experiments.supplement.Us.supplement-CIFAR10-Us --adjustment_literation -1 --parent_path log-supplement
```

## 📊 Datasets

### Dataset Downloads

#### MNIST
- [Training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- [Training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
- [Testing images](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
- [Testing labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

#### CIFAR10
- [CIFAR-10 Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

#### CIFAR100
- [CIFAR-100 Python](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

#### IMDB
- [IMDB Python](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

### Dataset Setup Instructions

1. Download the dataset files from the links above
2. Extract each dataset to its designated directory:
    - MNIST: Extract to `data/dataset/MNIST/`
    - CIFAR10: Extract to `data/dataset/CIFAR10/`
    - CIFAR100: Extract to `data/dataset/CIFAR100/`
    - IMDB：Extract to `data/dataset/IMDB/`
3. For IMDB Dataset Usage:
   - Download the NLTK tokenizers data (punkt and punkt_tab) from https://github.com/nltk/nltk_data/tree/gh-pages.
   - Extract the files from the packages/tokenizers directory in the downloaded repository.
   - Place the extracted files into /root/miniconda3/envs/py38/nltk_data/tokenizers/ to enable tokenizer functionality.

## 💡 Installation

You can install the package in development mode:

```bash
# Install package in development mode
pip install -e .
```

This allows you to modify the source code and have the changes take effect immediately without reinstalling.
