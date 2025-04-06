import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime
import os

from src.qd_rdfl.algorithms.GaleShapley import GaleShapley
from src.qd_rdfl.models.MNISTCNN import MNISTCNN, fine_tune_model, fine_tune_model_without_replace
from src.qd_rdfl.roles.CPC import CPC
from src.qd_rdfl.roles.DataOwner import DataOwner
from src.qd_rdfl.roles.ModelOwner import ModelOwner
from src.qd_rdfl.algorithms.Stackelberg import Stackelberg
from src.qd_rdfl.utils.UtilsMNIST import UtilsMNIST
from src.qd_rdfl.global_variable import global_minst_parent_path,Lambda,Rho,Alpha,Epsilon, adjustment_literation

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
    def find_project_root(current_dir):
        # 向上逐层查找，直到找到项目根目录
        while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 setup.py 文件
            current_dir = os.path.dirname(current_dir)
            if current_dir == '/':  # 避免在 Unix/Linux 系统中向上查找过多
                return None
        return current_dir

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 查找项目根目录
    project_root = find_project_root(current_dir)

    project_root = project_root.replace("\\", "/")

    if project_root:
        print("项目根目录:", project_root)
    else:
        print("未找到项目根目录")

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = f"{project_root}/data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = f"{project_root}/data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilsMNIST.load_mnist_dataset(train_images_path, train_labels_path)
    test_images, test_labels = UtilsMNIST.load_mnist_dataset(test_images_path, test_labels_path)

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有5个DataOwner

    # 切分数据
    UtilsMNIST.split_data_to_dataowners_evenly(dataowners, train_images, train_labels)

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
    UtilsMNIST.print_and_log(global_minst_parent_path, f"初始数据占MNIST的比例：{rate * 100}%")
    UtilsMNIST.print_and_log(global_minst_parent_path, "model initing...")

    def find_project_root(current_dir):
        # 向上逐层查找，直到找到项目根目录
        while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 setup.py 文件
            current_dir = os.path.dirname(current_dir)
            if current_dir == '/':  # 避免在 Unix/Linux 系统中向上查找过多
                return None
        return current_dir

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 查找项目根目录
    project_root = find_project_root(current_dir)

    project_root = project_root.replace("\\", "/")

    if project_root:
        print("项目根目录:", project_root)
    else:
        print("未找到项目根目录")

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = f"{project_root}/data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = f"{project_root}/data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilsMNIST.load_mnist_dataset(train_images_path, train_labels_path)
    test_images, test_labels = UtilsMNIST.load_mnist_dataset(test_images_path, test_labels_path)

    # 获取图像数量
    num_images = train_images.shape[0]
    # 计算需要选取的图像数量
    num_samples = int(num_images * rate)
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

    # 如果不存在初始化模型，就训练模型，如果存在，就加载到model中
    model_save_path = f"{project_root}/data/model/initial/mnist_cnn_initial_model"
    if os.path.exists(model_save_path):
        print(f"{model_save_path} 存在，加载初始化模型")
        model.load_model(model_save_path)
        model.save_model(f"{project_root}/data/model/mnist_cnn_model")
    else:
        print(f"{model_save_path} 不存在，初始化模型")
        model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                          model_save_path=model_save_path)
        model.save_model(f"{project_root}/data/model/mnist_cnn_model")

    return model

# ModelOwner发布任务， DataOwner计算数据质量（Dataowner自己计算）
def evaluate_data_quality(dataowners):
    """
    加噪声，模拟DataOwner的数据不好的情况
    :param dataowners:
    :param avg_f_list:
    :return:
    """

    # 评价数据质量
    for i, do in enumerate(dataowners):

        mse_scores = UtilsMNIST.evaluate_quality(do, metric="mse")
        snr_scores = UtilsMNIST.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0
        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # UtilsMNIST.print_and_log(global_minst_parent_path,f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilsMNIST.print_and_log(global_minst_parent_path, "DataOwners自行评估数据质量：")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
    UtilsMNIST.print_and_log(global_minst_parent_path,
                             f"归一化后的数据质量列表avg_f_list: {UtilsMNIST.normalize_list(avg_f_list)}")

    return UtilsMNIST.normalize_list(avg_f_list)


# ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
def calculate_optimal_payment_and_data(avg_f_list, last_xn_list):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list:
    :return:
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    eta_opt, x_opt, U_opt = Stackelberg.find_stackelberg_equilibrium(Alpha, np.array(avg_f_list), Lambda, Rho)

    UtilsMNIST.print_and_log(global_minst_parent_path, "Stackelberg均衡结果：")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"ModelOwner的最优Eta = {eta_opt:.4f}")
    xn_list = []
    for i, xi in enumerate(x_opt):
        UtilsMNIST.print_and_log(global_minst_parent_path, f"DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
        xn_list.append(xi)
    UtilsMNIST.print_and_log(global_minst_parent_path, f"每个DataOwner应该贡献数据比例 xn_list = {xn_list}")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"ModelOwner的最大效用 U(Eta) = {U_opt:.4f}")

    # 这里计算 U_Eta 和 U_qn
    U_Eta = U_opt
    U_qn = (eta_opt - Lambda * Rho * (sum(xn_list))) / N

    return UtilsMNIST.compare_elements(xn_list, last_xn_list), eta_opt, U_Eta, U_qn


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

    UtilsMNIST.print_and_log(global_minst_parent_path, f"ModelOwner的最优总支付：{best_Eta}")
    for i in range(len(xn_list)):
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"DataOwner{i + 1}的分配到的支付 ： {contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和CPC
def match_data_owners_to_cpc(xn_list, cpcs, dataowners):
    """
    匹配DataOwner和CPC
    :param xn_list:
    :param cpcs:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)

    preferences = GaleShapley.make_preferences(xn_list, cpcs, Rho, dataowners)

    # 调用Gale-Shapley算法
    matching = GaleShapley.gale_shapley(proposals, preferences)
    UtilsMNIST.print_and_log(global_minst_parent_path, matching)
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

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"DataOwner{dataowner_index + 1} 把数据交给 CPC{cpc_index + 1}")

        UtilsMNIST.dataowner_pass_data_to_cpc(dataowners[dataowner_index], cpcs[cpc_index], xn_list[dataowner_index])


# 使用CPC进行模型训练和全局模型的更新
def train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list, adjustment_literation):
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

    for item in matching.items():
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
        if len(cpcs[cpc_index].imgData) == 0:
            UtilsMNIST.print_and_log(global_minst_parent_path, "数据量为0，跳过此轮调整")
            continue

        train_loader = UtilsMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                     batch_size=64, shuffle=True)
        test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

        # 创建CNN模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTCNN(num_classes=10).to(device)


        def find_project_root(current_dir):
            # 向上逐层查找，直到找到项目根目录
            while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 setup.py 文件
                current_dir = os.path.dirname(current_dir)
                if current_dir == '/':  # 避免在 Unix/Linux 系统中向上查找过多
                    return None
            return current_dir

        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(current_file_path)

        # 查找项目根目录
        project_root = find_project_root(current_dir)

        project_root = project_root.replace("\\", "/")

        if project_root:
            print("项目根目录:", project_root)
        else:
            print("未找到项目根目录")

        fine_tune_model(model, train_loader, test_loader, num_epochs=5, device=str(device),
                        lr=1e-5, model_path=f"{project_root}/data/model/mnist_cnn_model")

    return UtilsMNIST.normalize_list(avg_f_list)


if __name__ == "__main__":
    UtilsMNIST.print_and_log(global_minst_parent_path,
                             f"**** {global_minst_parent_path}-MIX 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 记录第 adjustment_literation+1 轮的 U(Eta) 和 U(qn)/N
    U_Eta_list = []
    U_qn_list = []
    U_Eta_list_RANDOM = []
    U_qn_list_RANDOM = []
    U_Eta_list_FIX = []
    U_qn_list_FIX = []
    last_random_U_Eta = 0
    last_random_U_qn = 0
    last_fix_U_Eta = 0
    last_fix_U_qn = 0

    # 从这里开始进行不同数量客户端的循环 (前闭后开)
    for n in [9,19,29,39,49,59]:
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"========================= 客户端数量: {n + 1} =========================")

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- 定义参数值 ----------------------------------")
        Lambda, Rho, Alpha, Epsilon, N, M, SigmaM = define_parameters(Lambda=Lambda, Rho=Rho, Alpha=Alpha,
                                                                      Epsilon=Epsilon, M=n + 1, N=n + 1, SigmaM=[1] * (n + 1))
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, cpcs, test_images, test_labels = ready_for_task()
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        avg_f_list = []

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"----- 计算 DataOwner 的数据质量 -----")
        avg_f_list = evaluate_data_quality(dataowners)
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")



        """下面是三种方案的分别的效用计算"""
        """QD-RDFL"""

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N

        while True:
            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"========================= literation: {literation + 1} =========================")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
            last_xn_list = xn_list

            # 只有在调整轮次之后的轮次才记录
            if literation == adjustment_literation + 1:
                U_Eta_list.append(U_Eta)
                U_qn_list.append(U_qn)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list: {U_Eta_list}")
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list: {U_qn_list}")
                break

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsMNIST.print_and_log(global_minst_parent_path,
                                         f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            literation += 1



        """RANDOM"""
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- RANDOM：再次准备工作 ----------------------------------")
        _, modelowner, _, _, _ = ready_for_task()
        avg_f_list = []
        avg_f_list = evaluate_data_quality(dataowners)
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N
        while True:
            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"========================= literation: {literation + 1} =========================")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            # 这里获取到QD-RDFL方法的UEta，Uqn，接下来要随机一个Eta，然后根据公式求x_opt（一个集合），然后求Uqn
            xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
            last_xn_list = xn_list

            # 在调整轮次之后的轮次去做对比（这里未进行调整）
            if literation == adjustment_literation + 1:
                # 在0.5eta-eat的范围内取eta值
                random_Eta = random.uniform(0.5, 1) * best_Eta
                random_x_opt = Stackelberg._solve_followers(random_Eta, np.array(avg_f_list), Lambda, Rho)
                # 如果算不出最优解 就用上一轮的解
                if random_x_opt is None:
                    U_Eta_list_RANDOM.append(last_random_U_Eta)
                    U_qn_list_RANDOM.append(last_random_U_qn)
                    break
                random_xn_list = []
                for i, xi in enumerate(random_x_opt):
                    UtilsMNIST.print_and_log(global_minst_parent_path,
                                             f"random: DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
                    random_xn_list.append(xi)
                # random_xn_list = UtilsMNIST.compare_elements(random_xn_list, [0] * N)
                random_U_Eta = Stackelberg._leader_utility(random_Eta, Alpha, avg_f_list, random_xn_list)
                random_U_qn = (random_Eta - Lambda * Rho * (sum(random_xn_list))) / N

                # 添加进列表
                U_Eta_list_RANDOM.append(random_U_Eta)
                U_qn_list_RANDOM.append(random_U_qn)

                # 更新“上一值”
                last_random_U_Eta = random_U_Eta
                last_random_U_qn = random_U_qn

            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list_RANDOM: {U_Eta_list_RANDOM}")
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list_RANDOM: {U_qn_list_RANDOM}")
                break

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsMNIST.print_and_log(global_minst_parent_path,
                                         f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            literation += 1



        """FIX"""
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- FIX：再次准备工作 ----------------------------------")
        _, modelowner, _, _, _ = ready_for_task()
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N
        while True:
            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"========================= literation: {literation + 1} =========================")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            # 这里获取到QD-RDFL方法的UEta，Uqn，接下来要随机一个Eta，然后根据公式求x_opt（一个集合），然后求Uqn
            xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
            last_xn_list = xn_list

            # 在调整轮次之后的轮次去做对比（这里未进行调整）
            if literation == adjustment_literation + 1:
                # 在0.5eta-eat的范围内取eta值
                fix_Eta = 1
                fix_x_opt = Stackelberg._solve_followers(fix_Eta, np.array(avg_f_list), Lambda, Rho)
                if fix_x_opt is None:
                    U_Eta_list_FIX.append(last_fix_U_Eta)
                    U_qn_list_FIX.append(last_fix_U_qn)
                    break
                fix_xn_list = []
                for i, xi in enumerate(fix_x_opt):
                    UtilsMNIST.print_and_log(global_minst_parent_path,
                                             f"fix: DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
                    fix_xn_list.append(xi)
                # fix_xn_list = UtilsMNIST.compare_elements(fix_xn_list, [0] * N)
                fix_U_Eta = Stackelberg._leader_utility(fix_Eta, Alpha, avg_f_list, fix_xn_list)
                fix_U_qn = (fix_Eta - Lambda * Rho * (sum(fix_xn_list))) / N

                # 添加进列表
                U_Eta_list_FIX.append(fix_U_Eta)
                U_qn_list_FIX.append(fix_U_qn)

                # 更新“上一值”
                last_fix_U_Eta = fix_U_Eta
                last_fix_U_qn = fix_U_qn

            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list_FIX: {U_Eta_list_FIX}")
                UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list_FIX: {U_qn_list_FIX}")
                break

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsMNIST.print_and_log(global_minst_parent_path,
                                         f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            literation += 1



    UtilsMNIST.print_and_log(global_minst_parent_path, f"{N}:QD-RDFL的列表：")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list: {U_Eta_list}")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list: {U_qn_list}")

    UtilsMNIST.print_and_log(global_minst_parent_path, f"{N}:RANDOM的列表：")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list_RANDOM: {U_Eta_list_RANDOM}")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list: {U_qn_list_RANDOM}")

    UtilsMNIST.print_and_log(global_minst_parent_path, f"{N}:FIX最终的列表：")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_Eta_list_FIX: {U_Eta_list_FIX}")
    UtilsMNIST.print_and_log(global_minst_parent_path, f"U_qn_list_FIX: {U_qn_list_FIX}")