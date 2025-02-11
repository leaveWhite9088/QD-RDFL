# QD-RDFL

Quality-aware Dynamic Resources-decoupled FL 

## Model Framework

![framework](./data/imgs/framework.png)

In this paper, we focus on the approach of data assetization, that data do not circulate but the information does. We first propose a framework for resource-decoupled FL that involves model owners, data owners, and computing centers. Then, we design a Tripartite Stackelberg Model and theoretically analyze the Stackelberg-Nash equilibrium (SNE) for participants to optimize global utility. Next, we propose the Quality-aware Dynamic Resources-decoupled FL algorithm (QD-RDFL), in which we derive and solve the optimal strategies of all parties to achieve SNE using backward induction. We also design a dynamic optimization mechanism to improve the optimal strategy profile by evaluating the contribution of data quality from data owners to the global model during real training. Finally, our extensive experiments demonstrate that our method effectively encourages the linkage of the three parties involved, maximizing the global utility and value of data assets

## Requirements
```
torch==2.3.0
torchvision==0.18.1a0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
```

## Code Structures
```
.
│  .gitignore
│  global_variable.py
│  LICENSE
│  README.md
│
├─algorithm
│  │  GaleShapley.py
│  │  Stackelberg.py
│
├─data
│  │  .gitkeep
│  │
│  ├─dataset
│  │  ├─CIFAR10
│  │  │      batches.meta
│  │  │      data_batch_1
│  │  │      data_batch_2
│  │  │      data_batch_3
│  │  │      data_batch_4
│  │  │      data_batch_5
│  │  │      test_batch
│  │  │
│  │  ├─CIFAR100
│  │  │      meta
│  │  │      test
│  │  │      train
│  │  │
│  │  └─MNIST
│  │          t10k-images.idx3-ubyte
│  │          t10k-labels.idx1-ubyte
│  │          train-images.idx3-ubyte
│  │          train-labels.idx1-ubyte
│  │
│  ├─imgs
│  │  │  ablation-CIFAR10.png
│  │  │  ablation-CIFAR100.png
│  │  │  ablation-MNIST.png
│  │  │  comparison-CIFAR10-Uqn.png
│  │  │  comparison-CIFAR10-Us.png
│  │  │  comparison-CIFAR100-Uqn.png
│  │  │  comparison-CIFAR100-Us.png
│  │  │  comparison-MNIST-Uqn.png
│  │  │  comparison-MNIST-Us.png
│  │  │  optimized-imgs.zip
│  │  │  parameter_analysis-MNIST-Eta.png
│  │  │  parameter_analysis-MNIST-xn.png
│  │  │  supplement-CIFAR10-Us-100.png
│  │  │  supplement-MNIST-accurancy.png
│  │  │  supplement-MNIST-Un.png
│  │  │  supplement-MNIST-Us-100.png
│  │  │  supplement-MNIST-Us.png
│  │  │
│  │  └─optimized-imgs
│  │          plot-supplement-CIFAR10-Un.png
│  │          plot-supplement-CIFAR10-Us.png
│  │          plot-supplement-MNIST-Un.png
│  │          plot-supplement-MNIST-Us.png
│  │
│  ├─log
│  │  ├─log-ablation
│  │  │      log-ablation-CIFAR10.txt
│  │  │      log-ablation-CIFAR100.txt
│  │  │      log-ablation-MNIST.txt
│  │  │
│  │  ├─log-comparison
│  │  │      log-comparison-CIFAR10.txt
│  │  │      log-comparison-CIFAR100.txt
│  │  │      log-comparison-MNIST.txt
│  │  │
│  │  ├─log-main
│  │  │      log-CIFAR10.txt
│  │  │      log-CIFAR100.txt
│  │  │      log-MNIST.txt
│  │  │
│  │  ├─log-parameter_analysis
│  │  │      log-parameter_analysis-CIFAR10.txt
│  │  │      log-parameter_analysis-CIFAR100.txt
│  │  │      log-parameter_analysis-MNIST.txt
│  │  │
│  │  ├─log-supplement
│  │  │      log-supplement-CIFAR10.txt
│  │  │      log-supplement-MNIST.txt
│  │  │      log-supplemet-CIFAR100.txt
│  │  │
│  │  └─log-temp
│  │          log-temp-CIFAR10.txt
│  │          log-temp-CIFAR100.txt
│  │          log-temp-MNIST.txt
│  │
│  ├─model
│  │  │  cifar100_cnn_model
│  │  │  cifar10_cnn_model
│  │  │  mnist_cnn_model
│  │  │
│  │  └─initial
│  │          cifar100_cnn_initial_model
│  │          cifar10_cnn_initial_model
│  │          mnist_cnn_initial_model
│  │
│  └─saved
│      ├─parameter_analysis
│      │       parameter_analysis-Alpha.txt
│      │       parameter_analysis-Eta.txt
│      │       parameter_analysis-SigmaM.txt
│      │       parameter_analysis-xn.txt
│      │
│      ├─comparison
│      │       comparison-FIX.txt
│      │       comparison-MIX.txt
│      │       comparison-RANDOM.txt
│      │       comparison.txt
│      │
│      └─ablation
│              ablation-adjust.txt
│              ablation-noneadjust.txt
│
├─dataset
│  │  CIFAR100Dataset.py
│  │  CIFAR10Dataset.py
│  └─ MNISTDataset.py
│
├─experiment
│  ├─ablation
│  │  ├─adjust
│  │  │      ablation-CIFAR10-adjust.py
│  │  │      ablation-CIFAR100-adjust.py
│  │  │      ablation-MNIST-adjust.py
│  │  │
│  │  └─noneadjust
│  │          ablation-CIFAR10-noneadjust.py
│  │          ablation-CIFAR100-noneadjust.py
│  │          ablation-MNIST-noneadjust.py
│  │
│  ├─comparison
│  │  ├─FIX
│  │  │      comparison-CIFAR10-FIX.py
│  │  │      comparison-CIFAR100-FIX.py
│  │  │      comparison-MNIST-FIX.py
│  │  │
│  │  ├─MIX
│  │  │      comparison-CIFAR10-MIX.py
│  │  │      comparison-CIFAR100-MIX.py
│  │  │      comparison-MNIST-MIX.py
│  │  │
│  │  ├─QD-RDFL
│  │  │      comparison-CIFAR10.py
│  │  │      comparison-CIFAR100.py
│  │  │      comparison-MNIST.py
│  │  │
│  │  └─RANDOM
│  │          comparison-CIFAR10-RANDOM.py
│  │          comparison-CIFAR100-RANDOM.py
│  │          comparison-MNIST-RANDOM.py
│  │
│  ├─parameter_analysis
│  │  ├─Alpha
│  │  │      parameter_analysis-CIFAR10-Alpha.py
│  │  │      parameter_analysis-CIFAR100-Alpha.py
│  │  │      parameter_analysis-MNIST-Alpha.py
│  │  │
│  │  ├─Eta
│  │  │      parameter_analysis-MNIST-Eta.py
│  │  │
│  │  ├─Sigma
│  │  │      parameter_analysis-CIFAR10-Sigma.py
│  │  │      parameter_analysis-CIFAR100-Sigma.py
│  │  │      parameter_analysis-MNIST-Sigma.py
│  │  │
│  │  └─xn
│  │          parameter_analysis_MNIST-xn.py
│  │
│  └─supplement
│      ├─accurancy
│      │      supplement-CIFAR10-accurancy.py
│      │      supplement-MNIST-accurancy.py
│      │
│      ├─Un
│      │      supplement-CIFAR10-Un.py
│      │      supplement-MNIST-Un.py
│      │
│      └─Us
│              supplement-CIFAR10-Us.py
│              supplement-MNIST-Us.py
│
├─model
│  │  CIFAR100CNN.py
│  │  CIFAR10CNN.py
│  └─ MNISTCNN.py
│
├─plot
│  ├─ablation
│  │      plot-ablation-CIFAR10.py
│  │      plot-ablation-CIFAR100.py
│  │      plot-ablation-MNIST.py
│  │
│  ├─comparison
│  │      plot-comparison-CIFAR10-Uqn.py
│  │      plot-comparison-CIFAR10-Us.py
│  │      plot-comparison-CIFAR100-Uqn.py
│  │      plot-comparison-CIFAR100-Us.py
│  │      plot-comparison-MNIST-Uqn.py
│  │      plot-comparison-MNIST-Us.py
│  │
│  ├─parameter_analysis
│  │      plot-parameter_analysis-MNIST-Eta.py
│  │      plot-parameter_analysis-MNIST-xn.py
│  │
│  └─supplement
│          plot-supplement-CIFAR10-accurancy.py
│          plot-supplement-CIFAR10-Un.py
│          plot-supplement-CIFAR10-Us.py
│          plot-supplement-MNIST-accurancy.py
│          plot-supplement-MNIST-Un.py
│          plot-supplement-MNIST-Us.py
│
├─role
│  │  CPC.py
│  │  DataOwner.py
│  └─ ModelOwner.py
│
└─utils
   │  UtilsCIFAR10.py
   │  UtilsCIFAR100.py
   └─ UtilsMNIST.py
```

## Run Code

### ablation
- Ablation studies are time-consuming due to the need for training, with each experiment taking approximately 30 minutes.

#### adjust
- Observe the accuracy with dynamic adjustment.

```shell
python -m experiment.ablation.adjust.ablation-MNIST-adjust --adjustment_literation 2 
```
```shell
python -m experiment.ablation.adjust.ablation-CIFAR10-adjust --adjustment_literation 2 
```
```shell
python -m experiment.ablation.adjust.ablation-CIFAR100-adjust --adjustment_literation 2 
```
#### noneadjust
- Observe the accuracy without dynamic adjustment.

```shell
python -m experiment.ablation.noneadjust.ablation-MNIST-noneadjust --adjustment_literation 2 
```
```shell
python -m experiment.ablation.noneadjust.ablation-CIFAR10-noneadjust --adjustment_literation 2 
```
```shell
python -m experiment.ablation.noneadjust.ablation-CIFAR100-noneadjust --adjustment_literation 2 
```

### comparison
- The comparative experiment is time-consuming due to the need for training. Each run of the experiment takes about 30 minutes.

#### FIX
```shell
python -m experiment.comparison.FIX.comparison-MNIST-FIX --adjustment_literation 2
```
```shell
python -m experiment.comparison.FIX.comparison-CIFAR10-FIX --adjustment_literation 2
```
```shell
python -m experiment.comparison.FIX.comparison-CIFAR100-FIX --adjustment_literation 2
```

#### MIX
```shell
python -m experiment.comparison.MIX.comparison-MNIST-MIX --adjustment_literation 2
```
```shell
python -m experiment.comparison.MIX.comparison-CIFAR10-MIX --adjustment_literation 2
```
```shell
python -m experiment.comparison.MIX.comparison-CIFAR100-MIX --adjustment_literation 2
```

#### QD-RDFL
```shell
python -m experiment.comparison.QD-RDFL.comparison-MNIST-QD-RDFL --adjustment_literation 2
```
```shell
python -m experiment.comparison.QD-RDFL.comparison-CIFAR10-QD-RDFL --adjustment_literation 2
```
```shell
python -m experiment.comparison.QD-RDFL.comparison-CIFAR100-QD-RDFL --adjustment_literation 2
```

#### RANDOM
```shell
python -m experiment.comparison.RANDOM.comparison-MNIST-RANDOM --adjustment_literation 2
```
```shell
python -m experiment.comparison.RANDOM.comparison-CIFAR10-RANDOM --adjustment_literation 2
```
```shell
python -m experiment.comparison.RANDOM.comparison-CIFAR100-RANDOM --adjustment_literation 2
```

### parameter_analysis
- The experimental run speed of parameter analysis is very fast, and it can be completed within 30 seconds.

#### Alpha
- Investigate the changes in Us and the average Un under different values of Alpha.
```shell
python -m experiment.parameter_analysis.Alpha.parameter_analysis-MNIST-Alpha --adjustment_literation -1
```
```shell
python -m experiment.parameter_analysis.Alpha.parameter_analysis-CIFAR10-Alpha --adjustment_literation -1
```
```shell
python -m experiment.parameter_analysis.Alpha.parameter_analysis-CIFAR100-Alpha --adjustment_literation -1
```

#### Eta
- Investigate the changes in Us under different values of Eta.
```shell
python -m experiment.parameter_analysis.Eta.parameter_analysis-MNIST-Eta --adjustment_literation -1
```

#### Sigma
- Verify the impact of changes in SigmaM on the matching results.
```shell
python -m experiment.parameter_analysis.Sigma.parameter_analysis-MNIST-Sigma --adjustment_literation -1
```
```shell
python -m experiment.parameter_analysis.Sigma.parameter_analysis-CIFAR10-Sigma --adjustment_literation -1
```
```shell
python -m experiment.parameter_analysis.Sigma.parameter_analysis-CIFAR100-Sigma --adjustment_literation -1
```

#### xn
- Investigate the changes in the average Un under different values of xn.
```shell
python -m experiment.parameter_analysis.xn.parameter_analysis-MNIST-xn --adjustment_literation -1
```

### supplement

#### accurancy
```shell
python -m experiment.supplement.accurancy.supplement-MNIST-accurancy --adjustment_literation -1
```
```shell
python -m experiment.supplement.accurancy.supplement-CIFAR10-accurancy --adjustment_literation -1
```

#### Un
```shell
python -m experiment.supplement.Un.supplement-MNIST-Un --adjustment_literation -1
```
```shell
python -m experiment.supplement.Un.supplement-CIFAR10-Un --adjustment_literation -1
```

#### Us
```shell
python -m experiment.supplement.Us.supplement-MNIST-Us --adjustment_literation -1
```
```shell
python -m experiment.supplement.Us.supplement-CIFAR10-Us --adjustment_literation -1
```

## Dataset

### Get

#### MNIST：
- Training images：https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
- Training labels：https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
- Testing images：https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
- Testing labels：https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

#### CIFAR10：
- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

#### CIFAR100：
- https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

### Use
- MNIST: After extracting, place all the contents in the data/dataset/MNIST directory.
- CIFAR10: After extracting, place all the contents in the data/dataset/CIFAR10 directory.
- CIFAR100: After extracting, place all the contents in the data/dataset/CIFAR100 directory.