# QD-RDFL

Quality-aware Dynamic Resources-decoupled FL 

## 目录结构
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
│      ├─参数分析
│      │  │  parameter_analysis-Alpha.txt
│      │  │  parameter_analysis-Eta.txt
│      │  │  parameter_analysis-SigmaM.txt
│      │  │  parameter_analysis-xn.txt
│      │  │
│      │  └─参数分析 步长0.01
│      │          parameter_analysis-Eta.txt
│      │          parameter_analysis-xn.txt
│      │
│      ├─对比实验
│      │  │  comparison-FIX.txt
│      │  │  comparison-MIX.txt
│      │  │  comparison-RANDOM.txt
│      │  │  comparison.txt
│      │  │
│      │  └─动态调整+第三轮
│      │          comparison-FIX.txt
│      │          comparison-RANDOM.txt
│      │          comparison.txt
│      │
│      └─消融实验
│              ablation-adjust.txt
│              ablation-noneadjust.txt
│
├─dataset
│  │  CIFAR100Dataset.py
│  │  CIFAR10Dataset.py
│  │  MNISTDataset.py
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
│  │  │      parameter_analysis-CIFAR10-Eta.py
│  │  │      parameter_analysis-CIFAR100-Eta.py
│  │  │      parameter_analysis-MNIST-Eta.py
│  │  │
│  │  ├─Sigma
│  │  │      parameter_analysis-CIFAR10-Sigma.py
│  │  │      parameter_analysis-CIFAR100-Sigma.py
│  │  │      parameter_analysis-MNIST-Sigma.py
│  │  │
│  │  └─xn
│  │          parameter_analysis-CIFAR10-xn.py
│  │          parameter_analysis-CIFAR100-xn.py
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
│  │  MNISTCNN.py
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
│  │  ModelOwner.py
│
├─utils
│  │  UtilsCIFAR10.py
│  │  UtilsCIFAR100.py
│  │  UtilsMNIST.py
```

## 数据集
### 获取
- MNIST：https://pan.baidu.com/s/1jAPlVKLYamJn6I63GD6HDg?pwd=azq2 
- CIFAR10：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR100：https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
### 使用
- MNIST：解压后，把全部内容放置于data/dataset/MNIST目录下
- CIFAR10：解压后，把全部内容放置于data/dataset/CIFAR10目录下
- CIFAR100：解压后，把全部内容放置于data/dataset/CIFAR100目录下