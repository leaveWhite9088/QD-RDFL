import numpy as np
from algorithm.GaleShapley import GaleShapley
from algorithm.Stackelberg import Stackelberg
from model.CIFAR10CNN import CIFAR10CNN, fine_tune_model, fine_tune_model_without_replace
from role.CPC import CPC
from role.DataOwner import DataOwner
from role.ModelOwner import ModelOwner
from utils.UtilsCIFAR10 import UtilsCIFAR10
import random
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime
import os
from global_variable import global_cifar10_parent_path,Lambda,Rho,Alpha,Epsilon


# 定义参数值
def define_parameters(Lambda=1, Rho=1, Alpha=1, Epsilon=1, N=5, M=5, SigmaM=None):
    """
    定义参数值
    :param Lambda: 市场调整因子
    :param Rho: 单位数据训练费用
    :param Alpha: 模型质量调整参数
    :param Epsilon: 训练数据质量阈值
    :param N: DataOwner的数量
    :param M: CPC数量
    :param SigmaM: CPC的计算能力列表
    :return:
    """
    if SigmaM is None:
        SigmaM = [1] * M
    return Lambda, Rho, Alpha, Epsilon, N, M, SigmaM


# 为联邦学习任务做准备工作
def ready_for_task(Lambda, Rho, Alpha, Epsilon, N, M, SigmaM, data_dir):
    """
    准备联邦学习任务所需的数据和角色
    :param Lambda: 市场调整因子
    :param Rho: 单位数据训练费用
    :param Alpha: 模型质量调整参数
    :param Epsilon: 训练数据质量阈值
    :param N: DataOwner的数量
    :param M: CPC数量
    :param SigmaM: CPC的计算能力列表
    :param data_dir: CIFAR10数据集所在目录，包含批处理文件
    :return: dataowners, modelowner, cpcs, test_data, test_labels
    """
    # 加载CIFAR10数据集
    train_data, train_labels, test_data, test_labels = UtilsCIFAR10.load_cifar10_dataset(data_dir)

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有N个DataOwner

    # 切分数据
    UtilsCIFAR10.split_data_to_dataowners_with_large_gap(dataowners, train_data, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(Alpha, model=init_model())

    # 初始化CPC
    cpcs = [CPC(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, cpcs, test_data, test_labels


# modelowner的初始model
def init_model():
    """
    用于初始化一个模型给ModelOwner
    :return: 初始化后的模型
    """
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "初始化模型中...")

    # 假设初始比例为1% (rate=0.01)
    rate = 0.01
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"初始数据占CIFAR10的比例：{rate * 100}%")

    # 加载CIFAR10数据集
    data_dir = "../../../data/dataset/CIFAR10"  # CIFAR10批处理文件所在目录
    train_data, train_labels, _, _ = UtilsCIFAR10.load_cifar10_dataset(data_dir)

    # 获取图像数量
    num_images = train_data.shape[0]
    # 计算需要选取的图像数量
    num_samples = int(num_images * rate)
    # 随机生成索引
    indices = np.random.choice(num_images, num_samples, replace=False)
    # 使用随机索引选取数据
    selected_train_data = train_data[indices]
    selected_train_labels = train_labels[indices]

    # 创建训练和测试 DataLoader
    train_loader = UtilsCIFAR10.create_data_loader(selected_train_data, selected_train_labels, batch_size=128,
                                                   shuffle=True)
    # 这里的测试数据已在ready_for_task中加载，不需要重新加载

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果不存在初始化模型，就训练模型，如果存在，就加载到model中
    model_save_path = "../../../data/model/initial/cifar10_cnn_initial_model"
    if os.path.exists(model_save_path):
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"{model_save_path} 存在，加载初始化模型")
        model.load_model(model_save_path)
        model.save_model("../../../data/model/cifar10_cnn_model")
    else:
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"{model_save_path} 不存在，初始化模型")
        model.train_model(train_loader, criterion, optimizer, num_epochs=20, device=str(device),
                          model_save_path=model_save_path)
        model.save_model("../../../data/model/cifar10_cnn_model")

    # 加载完整的训练数据进行评估
    test_loader = UtilsCIFAR10.create_data_loader(train_data, train_labels, batch_size=128, shuffle=False)  # 使用全部数据进行测试

    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "初始化模型的准确率：")
    model.evaluate(test_loader, device=str(device))

    return model


# 给数据集添加噪声
def dataowner_add_noise(dataowners, rate):
    """
    给数据集添加噪声
    :param dataowners: DataOwner对象列表
    :param rate: 加噪（高斯噪声）的程度，初始程度在0-1之间
    :return:
    """
    # 第一次训练时：添加噪声，以1-MSE为fn
    for i, do in enumerate(dataowners):
        random_num = random.random() * rate
        UtilsCIFAR10.add_noise(do, noise_type="gaussian", severity=random_num)
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"DataOwner{i + 1}: noise random: {random_num}")


# ModelOwner发布任务， DataOwner计算数据质量（Dataowner自己计算）
def evaluate_data_quality(dataowners):
    """
    评价DataOwner的数据质量
    :param dataowners: DataOwner对象列表
    :return: 归一化后的数据质量列表
    """

    avg_f_list = []
    # 评价数据质量
    for i, do in enumerate(dataowners):

        mse_scores = UtilsCIFAR10.evaluate_quality(do, metric="mse")
        snr_scores = UtilsCIFAR10.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0
        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # 如果需要详细日志，可以取消下面注释
            # UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DataOwners自行评估数据质量：")
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
    normalized_avg_f_list = UtilsCIFAR10.normalize_list(avg_f_list)
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                               f"归一化后的数据质量列表avg_f_list: {normalized_avg_f_list}")

    return normalized_avg_f_list


# ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
def calculate_optimal_payment_and_data(avg_f_list, last_xn_list):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list: 数据质量列表
    :param last_xn_list: 上一次的x_n列表
    :return: xn_list, eta_opt, U_Eta, U_qn
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    eta_opt, x_opt, U_opt = Stackelberg.find_stackelberg_equilibrium(Alpha, np.array(avg_f_list), Lambda, Rho)

    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "Stackelberg均衡结果：")
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"ModelOwner的最优Eta = {eta_opt:.4f}")
    xn_list = []
    for i, xi in enumerate(x_opt):
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
        xn_list.append(xi)
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"每个DataOwner应该贡献数据比例 xn_list = {xn_list}")
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"ModelOwner的最大效用 U(Eta) = {U_opt:.4f}")

    # 这里计算 U_Eta 和 U_qn
    U_Eta = U_opt
    U_qn = (eta_opt - Lambda * Rho * sum(xn_list)) / N

    return UtilsCIFAR10.compare_elements(xn_list, last_xn_list), eta_opt, U_Eta, U_qn


# DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
def compute_contribution_rates(xn_list, avg_f_list, best_Eta):
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list: List of optimal data contributions from each DataOwner
    :param avg_f_list: List of normalized data quality scores for each DataOwner
    :param best_Eta: ModelOwner's optimal total payment
    :return: None
    """
    # 计算 qn （qn = xn * fn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]

    sum_qn = sum(contributions)

    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"ModelOwner的最优总支付：{best_Eta:.4f}")

    for i in range(len(xn_list)):
        if sum_qn == 0:
            payment = 0.0
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"DataOwner{i + 1}的分配到的支付 ： {payment:.4f} (sum_qn为0)")
        else:
            payment = (contributions[i] / sum_qn) * best_Eta
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"DataOwner{i + 1}的分配到的支付 ： {payment:.4f}")


# 匹配DataOwner和CPC
def match_data_owners_to_cpc(xn_list, cpcs):
    """
    匹配DataOwner和CPC
    :param xn_list: DataOwner的贡献比例列表
    :param cpcs: CPC对象列表
    :return: matching
    """
    preferences = GaleShapley.make_preferences(xn_list, cpcs, Rho)
    proposals = GaleShapley.make_proposals(SigmaM, N)

    # 调用Gale-Shapley算法
    matching = GaleShapley.gale_shapley(proposals, preferences)
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"Matching结果: {matching}")
    return matching


# DataOwner向CPC提交数据
def submit_data_to_cpc(matching, dataowners, cpcs, xn_list):
    """
    DataOwner按照xn_list中约定的比例向CPC提交数据
    :param matching: GaleShapley匹配结果（字典形式，键为DataOwner，值为CPC）
    :param dataowners: DataOwner对象列表
    :param cpcs: CPC对象列表
    :param xn_list: 需要提交的数据的比例列表
    :return: None
    """
    for dataowner_name, cpc_name in matching.items():
        # 使用正则表达式匹配字符串末尾的数字
        dataowner_match = re.search(r'\d+$', dataowner_name)
        dataowner_index = int(dataowner_match.group()) - 1 if dataowner_match else None

        cpc_match = re.search(r'\d+$', cpc_name)
        cpc_index = int(cpc_match.group()) - 1 if cpc_match else None

        if dataowner_index is not None and cpc_index is not None:
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"DataOwner{dataowner_index + 1} 把数据交给 CPC{cpc_index + 1}")

            UtilsCIFAR10.dataowner_pass_data_to_cpc(dataowners[dataowner_index],
                                                    cpcs[cpc_index],
                                                    xn_list[dataowner_index])
        else:
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"匹配失败：{dataowner_name} 或 {cpc_name} 的索引无法解析。")


# 使用CPC进行模型训练和全局模型的更新
def train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list, adjustment_literation, N):
    """
    使用CPC进行模型训练和全局模型的更新
    :param matching: GaleShapley匹配结果（字典形式，键为DataOwner，值为CPC）
    :param cpcs: CPC对象列表
    :param test_images: 测试图像数据
    :param test_labels: 测试图像标签
    :param literation: 当前迭代次数
    :param avg_f_list: 数据质量评分列表
    :param adjustment_literation: 需要进行数据质量评估和调整的迭代次数
    :param N: DataOwner的数量
    :return: 归一化后的数据质量评分列表
    """

    # 指定轮次的时候要评估数据质量，其余轮次直接训练即可
    if literation == adjustment_literation:
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "重新调整fn，进而调整xn、Eta")
        avg_f_list = [0] * N
        for item in matching.items():
            # 使用正则表达式匹配字符串末尾的数字
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1 if dataowner_match else None
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1 if cpc_match else None

            if dataowner_index is None or cpc_index is None:
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                           f"匹配失败：{item[0]} 或 {item[1]} 的索引无法解析。")
                continue

            cpc_data_len = len(cpcs[cpc_index].imgData)
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{cpc_data_len} :")
            if cpc_data_len == 0:
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "数据量为0，跳过此轮评估")
                continue

            train_loader = UtilsCIFAR10.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                           batch_size=64, shuffle=True)
            test_loader = UtilsCIFAR10.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 创建CNN模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CIFAR10CNN(num_classes=10).to(device)

            unitDataLossDiff = fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=5,
                                                               device=str(device),
                                                               lr=1e-5, model_path="../../../data/model/cifar10_cnn_model")
            avg_f_list[dataowner_index] = unitDataLossDiff

        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "经过服务器调节后的真实数据质量：")
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
        normalized_avg_f_list = UtilsCIFAR10.normalize_list(avg_f_list)
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                   f"归一化后的数据质量列表avg_f_list:{normalized_avg_f_list}")

    for item in matching.items():
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1 if dataowner_match else None
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1 if cpc_match else None

        if dataowner_index is None or cpc_index is None:
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"匹配失败：{item[0]} 或 {item[1]} 的索引无法解析。")
            continue

        cpc_data_len = len(cpcs[cpc_index].imgData)
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                   f"CPC{cpc_index + 1} 调整模型中, 本轮训练的数据量为：{cpc_data_len} :")
        if cpc_data_len == 0:
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "数据量为0，跳过此轮调整")
            continue

        train_loader = UtilsCIFAR10.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                       batch_size=64, shuffle=True)
        test_loader = UtilsCIFAR10.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

        # 创建CNN模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CIFAR10CNN(num_classes=10).to(device)

        fine_tune_model(model, train_loader, test_loader, num_epochs=5, device=str(device),
                        lr=1e-5, model_path="../../../data/model/cifar10_cnn_model")

    return UtilsCIFAR10.normalize_list(avg_f_list)


if __name__ == "__main__":
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                               f"**** {global_cifar10_parent_path} 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 记录第 adjustment_literation+1 轮的 U(Eta) 和 U(qn)/N
    U_Eta_list = []
    U_qn_list = []

    # 从这里开始进行不同数量客户端的循环 (前闭后开)
    for n in range(1, 101):
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                   f"========================= 客户端数量: {n + 1} =========================")

        UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                   "---------------------------------- 定义参数值 ----------------------------------")
        Lambda, Rho, Alpha, Epsilon, N, M, SigmaM = define_parameters(Lambda=Lambda, Rho=Rho, Alpha=Alpha,
                                                                      Epsilon=Epsilon,  M=n + 1, N=n + 1, SigmaM=[1] * (n + 1))
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

        UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                   "---------------------------------- 准备工作 ----------------------------------")
        data_dir = "../../../data/dataset/CIFAR10"  # CIFAR10批处理文件所在目录
        dataowners, modelowner, cpcs, test_data, test_labels = ready_for_task(Lambda, Rho, Alpha, Epsilon, N, M, SigmaM,
                                                                              data_dir)
        UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = 1  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        avg_f_list = []
        last_xn_list = [0] * N
        while True:
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"========================= literation: {literation + 1} =========================")

            # DataOwner自己报数据质量的机会只有一次
            if literation == 0:
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                           f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
                dataowner_add_noise(dataowners, 0.1)
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

                UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                           f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
                avg_f_list = evaluate_data_quality(dataowners)
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
            last_xn_list = xn_list

            # 只有在调整轮次之后的轮次才记录
            if literation == adjustment_literation + 1:
                U_Eta_list.append(U_Eta)
                U_qn_list.append(U_qn)
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"U_Eta_list: {U_Eta_list}")
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"U_qn_list: {U_qn_list}")
                break

            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                           f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs)
                UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            UtilsCIFAR10.print_and_log(global_cifar10_parent_path,
                                       f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_data, test_labels, literation, avg_f_list,
                                              adjustment_literation, N)
            UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "DONE")

            literation += 1

    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, "最终的列表：")
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"U_Eta_list: {U_Eta_list}")
    UtilsCIFAR10.print_and_log(global_cifar10_parent_path, f"U_qn_list: {U_qn_list}")
