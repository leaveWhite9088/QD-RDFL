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
import os
from global_variable import global_minst_parent_path, Lambda, Rho, Alpha, Epsilon, adjustment_literation


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


# 给数据集添加噪声
def dataowner_add_noise(dataowners, rate):
    """
    给数据集添加噪声
    :param dataowners:
    :param rate: 加噪（高斯噪声）的程度，初始程度在0-1之间
    :return:
    """
    # 第一次训练时：添加噪声，以1-MSE为fn
    for i, do in enumerate(dataowners):
        random_num = random.random() * rate
        UtilsMNIST.add_noise(do, severity=random_num)
        UtilsMNIST.print_and_log(global_minst_parent_path, f"DataOwner{i + 1}: noise random: {random_num}")


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

    print(f"preferences:{preferences}")

    # 调用Gale-Shapley算法
    matching = GaleShapley.gale_shapley(proposals, preferences)

    # 打印匹配结果
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

    # 指定轮次的时候要评估数据质量, 其余轮次直接训练即可
    if literation == adjustment_literation:
        UtilsMNIST.print_and_log(global_minst_parent_path, "重新调整fn，进而调整xn、Eta")
        avg_f_list = [0] * N
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilsMNIST.print_and_log(global_minst_parent_path, "数据量为0，跳过此轮评估")
                continue

            train_loader = UtilsMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                         batch_size=64, shuffle=True)
            test_loader = UtilsMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 创建CNN模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MNISTCNN(num_classes=10).to(device)

            unitDataLossDiff = fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=5,
                                                               device=str(device),
                                                               lr=1e-5,
                                                               model_path="../../../data/model/mnist_cnn_model")
            avg_f_list[dataowner_index] = unitDataLossDiff

        UtilsMNIST.print_and_log(global_minst_parent_path, "经过服务器调节后的真实数据质量：")
        UtilsMNIST.print_and_log(global_minst_parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"归一化后的数据质量列表avg_f_list:{UtilsMNIST.normalize_list(avg_f_list)}")

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

        fine_tune_model(model, train_loader, test_loader, num_epochs=5, device=str(device),
                        lr=1e-5, model_path="../../../data/model/mnist_cnn_model")

    return UtilsMNIST.normalize_list(avg_f_list)


if __name__ == "__main__":
    UtilsMNIST.print_and_log(global_minst_parent_path,
                             f"**** {global_minst_parent_path} 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 从这里开始进行不同数量客户端的循环 (前闭后开)
    for n in [9]:
        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"========================= 客户端数量: {n + 1} =========================")

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- 定义参数值 ----------------------------------")
        Lambda, Rho, Alpha, Epsilon, N, M, SigmaM = define_parameters(Lambda=Lambda, Rho=Rho, Alpha=Alpha,
                                                                      Epsilon=Epsilon, M=n + 1, N=n + 1,
                                                                      SigmaM=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                                              0.9, 1.0])
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 "---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, cpcs, test_images, test_labels = ready_for_task()
        for cpc in cpcs:
            print(cpc.SigmaM)
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        avg_f_list = []
        last_xn_list = [0] * N

        # DataOwner自己报数据质量的机会只有一次
        if literation == 0:
            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
            dataowner_add_noise(dataowners, 0.1)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

            UtilsMNIST.print_and_log(global_minst_parent_path,
                                     f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
            avg_f_list = evaluate_data_quality(dataowners)
            UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
        xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
        last_xn_list = xn_list

        UtilsMNIST.print_and_log(global_minst_parent_path,
                                 f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
        compute_contribution_rates(xn_list, avg_f_list, best_Eta)
        UtilsMNIST.print_and_log(global_minst_parent_path, "DONE")

        # 数据量列表
        data_volume_list = []
        for do in dataowners:
            data_volume_list.append(len(do.imgData))
        data_volume_list = [x * y for x, y in zip(data_volume_list, xn_list)]

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

        # 改变：从Eta复制过来，加了下面的这段代码，修改了range，SigmaM，删除了while,训练代码和提前终止

        new_Um_list = []

        sumdm = sum(xn_list)
        sumxn = sum(xn_list)

        # 这里再次计算dm*并归一化
        bestDm_list = GaleShapley.nash_equilibrium(cpcs, Rho, sumxn, xn_list)

        min_value = min(bestDm_list)
        max_value = max(bestDm_list)
        minVal = min(data_volume_list)
        maxVal = max(data_volume_list)

        normalized_lst = [minVal + (x - min_value) * (maxVal - minVal) / (max_value - min_value) for x in
                          bestDm_list]

        # 这里对matching进行处理，获取需要的Um，data，Sigma
        for key_do, val_cpc in matching.items():
            do_number = int(re.findall(r'\d+', key_do)[-1])
            cpc_number = int(re.findall(r'\d+', val_cpc)[-1])

            # 获取 cpc
            temp_cpc = cpcs[cpc_number - 1]

            # 计算 Um
            Um = temp_cpc.cal_cpc_utility(Rho, sumdm, sumxn, xn_list[do_number - 1])

            # 同时要记录数据量
            data_volume = len(temp_cpc.imgData)

            # DO，CPC，SigmaM，xm，Um
            # TODO DO，CPC，SigmaM，dm，xm，Um
            temp_tuple = (key_do, val_cpc, SigmaM[cpc_number - 1], normalized_lst[cpc_number - 1], data_volume, Um)

            new_Um_list.append(temp_tuple)

        if literation > adjustment_literation + 1:
            break

    UtilsMNIST.print_and_log(global_minst_parent_path, "最终Um的列表：")
    UtilsMNIST.print_and_log(global_minst_parent_path, new_Um_list)
    for temp_print_item in new_Um_list:
        UtilsMNIST.print_and_log(global_minst_parent_path, temp_print_item)
