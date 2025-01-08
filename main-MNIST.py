import numpy as np

from algorithm.GaleShapley import GaleShapley
from model.MNISTCNN import MNISTCNN, fine_tune_model, fine_tune_model_without_replace
from role.CPC import CPC
from role.DataOwner import DataOwner
from role.ModelOwner import ModelOwner
from algorithm.Stackelberg import Stackelberg
from utils.UtilsMNIST import UtilsMNIST
import random
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime


# 定义参数值
def define_parameters(Lambda=1, Rho=1, Alpha=1, Epsilon=1, N=5, M=5, SigmaM=[1, 1, 1, 1, 1]):
    """
    定义参数值
    :param Lambda: 市场调整因子
    :param Rho: 单位数据训练费用
    :param Alpha: 模型质量调整参数
    :param Epsilon: 训练数据质量阈值
    :param N: DataOwner的数量
    :param M: CPC数量
    :param SigmaM: CPC的计算呢能力
    :return:
    """
    return Lambda, Rho, Alpha, Epsilon, N, M, SigmaM


# 为联邦学习任务做准备工作
def ready_for_task():
    train_images_path = "./data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = "./data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = "./data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = "./data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilsMNIST.load_mnist_dataset(train_images_path, train_labels_path)
    test_images, test_labels = UtilsMNIST.load_mnist_dataset(test_images_path, test_labels_path)

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有5个DataOwner

    # 切分数据
    UtilsMNIST.split_data_to_dataowners_with_large_gap(dataowners, train_images, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(Alpha, model=init_model(0.001))

    # 初始化CPC
    cpcs = [CPC(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, cpcs, test_images, test_labels


# modelowner的初始model
def init_model(rate):
    """
    用于初始化一个模型给modeloowner
    :param rate: 初始数据占MNIST的比例
    :return:
    """
    UtilsMNIST.print_and_log(f"初始数据占MNIST的比例：{rate * 100}%")
    UtilsMNIST.print_and_log("model initing...")

    train_images_path = "./data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = "./data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = "./data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = "./data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilsMNIST.load_mnist_dataset(train_images_path, train_labels_path)
    test_images, test_labels = UtilsMNIST.load_mnist_dataset(test_images_path, test_labels_path)

    # 获取图像数量
    num_images = train_images.shape[0]
    # 计算需要选取的图像数量
    num_samples = num_images // int(1.0 / rate)
    # 随机生成索引
    indices = np.random.choice(num_images, num_samples, replace=False)
    # 使用随机索引选取数据
    train_labels = train_labels[indices]
    train_images = train_images[indices]

    train_loader = UtilsMNIST.create_data_loader(train_images, train_labels, batch_size=64, shuffle=True)
    test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model_save_path = "./data/model/mnist_cnn_test"
    model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                      model_save_path=model_save_path)

    UtilsMNIST.print_and_log("初始化模型的准确率：")
    model.evaluate(test_loader, device=str(device))

    return model


# ModelOwner发布任务， DataOwner计算数据质量
def evaluate_data_quality(dataowners, literation, rate, avg_f_list, printTimes):
    """
    加噪声，模拟DataOwner的数据不好的情况
    :param dataowners:
    :param rate: 加噪（高斯噪声）的程度，初始程度在0-1之间
    :param avg_f_list:
    :param UtilsMNIST.printTimes: 控制打印的图片的质量的次数
    :return:
    """

    if literation == 0:
        # 第一次训练时：添加噪声，以1-MSE为fn
        for i, do in enumerate(dataowners):
            random_num = random.random() / int(1.0 / rate)
            UtilsMNIST.add_noise(do, severity=random_num)
            UtilsMNIST.print_and_log(f"DataOwner{i + 1}: noise random: {random_num}")

        # 评价数据质量
        for i, do in enumerate(dataowners):

            mse_scores = UtilsMNIST.evaluate_quality(do, metric="mse")
            snr_scores = UtilsMNIST.evaluate_quality(do, metric="snr")

            # 输出每张图像的质量得分
            cnt = 0
            avg_f = 0
            for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
                UtilsMNIST.print_and_log(f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
                cnt += 1
                avg_f += mse
                if cnt >= printTimes:
                    break
            avg_f /= printTimes
            avg_f_list.append(1 - avg_f)

        return avg_f_list

    if literation >= 1:
        # 第二次的avg_f_list应该就是上一轮的loss差，所以这里直接返回
        return avg_f_list


# ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
def calculate_optimal_payment_and_data(avg_f_list):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list:
    :return:
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    eta_opt, x_opt, U_opt = Stackelberg.find_stackelberg_equilibrium(Alpha, np.array(avg_f_list), Lambda, Rho)

    UtilsMNIST.print_and_log("Stackelberg均衡结果：")
    UtilsMNIST.print_and_log(f"ModelOwner的最优Eta = {eta_opt:.4f}")
    xn_list = []
    for i, xi in enumerate(x_opt):
        UtilsMNIST.print_and_log(f"DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
        xn_list.append(xi)
    UtilsMNIST.print_and_log(f"每个DataOwner应该贡献数据比例 xn_list = {xn_list}")
    UtilsMNIST.print_and_log(f"ModelOwner的最大效用 U(Eta) = {U_opt:.4f}")

    return xn_list, eta_opt


# DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
def compute_contribution_rates(xn_list, avg_f_list, best_Eta):
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list:
    :param avg_f_list:
    :param best_Eta:
    :return:
    """
    # 计算qn （qn = xn*fn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]

    sum_qn = sum(contributions)

    UtilsMNIST.print_and_log(f"ModelOwner的最优总支付：{best_Eta}")
    for i in range(len(xn_list)):
        UtilsMNIST.print_and_log(f"DataOwner{i + 1}的分配到的支付 ： {contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和CPC
def match_data_owners_to_cpc(xn_list, cpcs):
    """
    匹配DataOwner和CPC
    :param xn_list:
    :param cpcs:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, 5)

    preferences = GaleShapley.make_preferences(xn_list, cpcs, Rho)

    # 调用Gale-Shapley算法
    matching = GaleShapley.gale_shapley(proposals, preferences)
    UtilsMNIST.print_and_log(matching)
    return matching


# DataOwner向CPC提交数据
def submit_data_to_cpc(matching, dataowners, cpcs, xn_list):
    """
    DataOwner按照xn_list中约定的比例向CPC提交数据
    :param matching:
    :param dataowners:
    :param cpcs:
    :param xn_list: 需要提交的数据的比例
    :return:
    """
    for item in matching.items():
        # 使用正则表达式匹配字符串末尾的数字
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilsMNIST.print_and_log(f"DataOwner{dataowner_index + 1} 把数据交给 CPC{cpc_index + 1}")

        UtilsMNIST.dataowner_pass_data_to_cpc(dataowners[dataowner_index], cpcs[cpc_index], xn_list[dataowner_index])


# 使用CPC进行模型训练和全局模型的更新
def train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list):
    """
    使用CPC进行模型训练和全局模型的更新
    :param matching:
    :param dataowners:
    :param test_images:
    :param test_labels:
    :param literation:训练的伦茨
    :param avg_f_list:fn的列表
    :return: 第二轮要使用的fn的列表
    """

    # 第一次训练的时候要评估数据质量
    if literation == 0:
        avg_f_list = [0] * N
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilsMNIST.print_and_log(
                f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilsMNIST.print_and_log("数据量为0，跳过此轮评估")
                continue

            train_loader = UtilsMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                         batch_size=64, shuffle=True)
            test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 创建CNN模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MNISTCNN(num_classes=10).to(device)

            diffloss = fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=5, device='cpu',
                                                       lr=1e-5, model_path="./data/model/mnist_cnn_test")
            avg_f_list[dataowner_index] = diffloss

        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilsMNIST.print_and_log(
                f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilsMNIST.print_and_log("数据量为0，跳过此轮调整")
                continue

            train_loader = UtilsMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                         batch_size=64, shuffle=True)
            test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 创建CNN模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MNISTCNN(num_classes=10).to(device)

            fine_tune_model(model, train_loader, test_loader, num_epochs=5, device='cpu',
                            lr=1e-5, model_path="./data/model/mnist_cnn_test")

        return avg_f_list

    # 第二次直接训练即可
    if literation >= 1:
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilsMNIST.print_and_log(
                f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilsMNIST.print_and_log("数据量为0，跳过此轮调整")
                continue

            train_loader = UtilsMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                         batch_size=64, shuffle=True)
            test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 创建CNN模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MNISTCNN(num_classes=10).to(device)

            fine_tune_model(model, train_loader, test_loader, num_epochs=5, device='cpu',
                            lr=1e-5, model_path="./data/model/mnist_cnn_test")

        # 不需要调整，直接返回即可
        return avg_f_list


if __name__ == "__main__":
    UtilsMNIST.print_and_log(
        f"*************************** 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ***************************")

    UtilsMNIST.print_and_log("---------------------------------- 定义参数值 ----------------------------------")
    Lambda, Rho, Alpha, Epsilon, N, M, SigmaM = define_parameters()
    UtilsMNIST.print_and_log("DONE")

    UtilsMNIST.print_and_log("---------------------------------- 准备工作 ----------------------------------")
    dataowners, modelowner, cpcs, test_images, test_labels = ready_for_task()
    UtilsMNIST.print_and_log("DONE")

    literation = 0  # 迭代次数
    avg_f_list = []
    while True:
        UtilsMNIST.print_and_log(
            f"==================================== literation: {literation + 1} ====================================")

        UtilsMNIST.print_and_log(
            "---------------------------------- 计算 DataOwner 的数据质量 ----------------------------------")
        avg_f_list = evaluate_data_quality(dataowners, literation, 0.1, avg_f_list, 1)
        UtilsMNIST.print_and_log("DONE")

        UtilsMNIST.print_and_log(
            "---------------------------------- 计算 ModelOwner 总体支付和 DataOwners 最优数据量 ----------------------------------")
        xn_list, best_Eta = calculate_optimal_payment_and_data(avg_f_list)
        UtilsMNIST.print_and_log("DONE")

        UtilsMNIST.print_and_log(
            "---------------------------------- DataOwner 分配 ModelOwner 的支付 ----------------------------------")
        compute_contribution_rates(xn_list, avg_f_list, best_Eta)
        UtilsMNIST.print_and_log("DONE")

        UtilsMNIST.print_and_log(
            "---------------------------------- 匹配 DataOwner 和 CPC ----------------------------------")
        matching = match_data_owners_to_cpc(xn_list, cpcs)
        UtilsMNIST.print_and_log("DONE")

        UtilsMNIST.print_and_log(
            "---------------------------------- DataOwner 向 CPC 提交数据 ----------------------------------")
        submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
        UtilsMNIST.print_and_log("DONE")

        UtilsMNIST.print_and_log("---------------------------------- 模型训练 ----------------------------------")
        avg_f_list = train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list)
        UtilsMNIST.print_and_log("DONE")

        literation += 1
        if literation >= 2:
            break
