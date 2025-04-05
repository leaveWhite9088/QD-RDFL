"""全局变量"""

"""================================= 路径参数 ================================="""

# 可选：log-parameter_analysis,log-comparison,log-ablation,log-supplement

global_minst_parent_path = "log-parameter_analysis"
global_cifar10_parent_path = "log-parameter_analysis"
global_cifar100_parent_path = "log-parameter_analysis"

"""================================= 超参数 ================================="""

Lambda = 1
Rho = 1
Alpha = 5
Epsilon = 1


"""================================= 其他参数 ================================="""

# TODO 这里要通过命令行参数修改值

# parameter_analysis-MNIST-xn.py
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Parameter Analysis for MNIST with XN.")

# 添加命令行参数
parser.add_argument('--adjustment_literation', type=float, default=-1, help="adjustment_literation")

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数的值
adjustment_literation = args.adjustment_literation
