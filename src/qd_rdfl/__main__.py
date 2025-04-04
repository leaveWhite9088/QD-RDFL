"""
主入口点文件，这样可以通过 python -m qd_rdfl 来运行项目
"""
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='QD-RDFL: 数据资产化去耦联邦学习框架')
    parser.add_argument('--mode', type=str, default='help', 
                        help='运行模式: help, run_experiment')
    parser.add_argument('--experiment', type=str, default=None,
                        help='要运行的实验类型')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='数据集类型: MNIST, CIFAR10, CIFAR100')
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        print("使用方法: python -m qd_rdfl --mode [mode] --experiment [experiment] --dataset [dataset]")
        print("可用模式:")
        print("  help: 显示帮助信息")
        print("  run_experiment: 运行实验")
        print("可用实验:")
        print("  comparison: 运行比较实验")
        print("  supplement: 运行补充实验")
        print("  parameter_analysis: 运行参数分析实验")
        print("  ablation: 运行消融实验")
        print("可用数据集:")
        print("  MNIST: MNIST数据集")
        print("  CIFAR10: CIFAR10数据集")
        print("  CIFAR100: CIFAR100数据集")
    
    elif args.mode == 'run_experiment':
        if args.experiment is None:
            print("请指定要运行的实验类型")
            return
        
        # 根据args.experiment和args.dataset调用相应的实验
        print(f"运行实验: {args.experiment}, 数据集: {args.dataset}")
        # TODO: 实现实验的具体调用逻辑
    
    else:
        print(f"未知模式: {args.mode}")
        print("使用 --mode help 查看帮助信息")

if __name__ == "__main__":
    main() 