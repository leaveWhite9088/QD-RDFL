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
import os

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Parameter Analysis for MNIST with XN.")

# 添加命令行参数
parser.add_argument('--adjustment_literation', type=float, default=0.01, help="adjustment_literation")

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数的值
adjustment_literation = args.adjustment_literation

def find_project_root(current_dir):
    # 添加Windows路径检测 
    if os.name == 'nt':  # Windows系统
        drive = os.path.splitdrive(current_dir)[0] + '\\'
        if current_dir == drive:  # 已到达驱动器根目录
            return None
    elif current_dir == '/':  # Unix/Linux系统的根目录
        return None
        
    # 检查README.md是否存在
    if os.path.exists(os.path.join(current_dir, 'README.md')):
        return current_dir
    else:
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # 防止无限循环
            return None
        return find_project_root(parent_dir)
