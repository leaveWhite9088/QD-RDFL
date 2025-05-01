import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime
import os
import pickle

from src.qd_rdfl.algorithms.GaleShapley import GaleShapley
from src.qd_rdfl.models.IMDBTextCNN import IMDBTextCNN, fine_tune_model, fine_tune_model_without_replace
from src.qd_rdfl.roles.CPC import CPC
from src.qd_rdfl.roles.DataOwner import DataOwner
from src.qd_rdfl.roles.ModelOwner import ModelOwner
from src.qd_rdfl.algorithms.Stackelberg import Stackelberg
from src.qd_rdfl.utils.UtilsIMDB import UtilsIMDB
from src.qd_rdfl.global_variable import Lambda, Rho, Alpha, Epsilon, adjustment_literation, global_imdb_parent_path


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
    :param SigmaM: CPC的计算能力
    :return:
    """
    return Lambda, Rho, Alpha, Epsilon, N, M, SigmaM


# 为联邦学习任务做准备工作
def ready_for_task():
    def find_project_root(current_dir):
        # 向上逐层查找，直到找到项目根目录
        while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 README.md 文件
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

    # 加载IMDB数据集
    data_dir = f"{project_root}/data/dataset/IMDB"

    # 加载训练数据
    train_texts, train_labels = UtilsIMDB.load_imdb_dataset(data_dir, mode='train')
    test_texts, test_labels = UtilsIMDB.load_imdb_dataset(data_dir, mode='test')

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有N个DataOwner

    # 切分数据
    UtilsIMDB.split_data_to_dataowners_evenly(dataowners, train_texts, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(Alpha, model=init_model(0.1))

    # 初始化CPC
    cpcs = [CPC(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, cpcs, test_texts, test_labels


# modelowner的初始model
def init_model(rate):
    """
    用于初始化一个模型给modelowner
    :param rate: 初始数据占IMDB的比例
    :return:
    """
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"初始数据占IMDB的比例：{rate * 100}%")
    UtilsIMDB.print_and_log(global_imdb_parent_path, "model initing...")

    def find_project_root(current_dir):
        # 向上逐层查找，直到找到项目根目录
        while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 README.md 文件
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

    data_dir = f"{project_root}/data/dataset/IMDB"

    # 加载训练数据和测试数据
    train_texts, train_labels = UtilsIMDB.load_imdb_dataset(data_dir, mode='train')
    test_texts, test_labels = UtilsIMDB.load_imdb_dataset(data_dir, mode='test')

    # 获取样本数量
    num_samples = len(train_texts)
    # 计算需要选取的样本数量
    num_to_use = int(num_samples * rate)
    # 随机生成索引
    indices = np.random.choice(num_samples, num_to_use, replace=False)
    # 使用随机索引选取数据
    selected_train_texts = [train_texts[i] for i in indices]
    selected_train_labels = [train_labels[i] for i in indices]

    # 模型保存路径
    model_save_path = f"{project_root}/data/model/initial/imdb_textcnn_initial_model"
    # 词汇表保存路径
    vocab_path = f"{project_root}/data/model/imdb_vocab.pkl"

    # 如果已存在预训练模型和词汇表，直接加载
    if os.path.exists(model_save_path) and os.path.exists(vocab_path):
        print(f"{model_save_path} 存在，加载初始化模型和词汇表")
        # 加载保存的词汇表
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IMDBTextCNN(
            vocab_size=len(vocab),
            embed_dim=100,
            filter_sizes=[3, 4, 5],
            num_filters=100,
            num_classes=2,
            dropout=0.5
        ).to(device)
        # 加载预训练权重
        model.load_model(model_save_path)
        model.save_model(f"{project_root}/data/model/imdb_textcnn_model")
    else:
        print(f"{model_save_path} 不存在，初始化模型")
        # 创建新词汇表
        vocab = UtilsIMDB.create_vocab(selected_train_texts, vocab_size=10000)
        # 创建数据加载器
        train_loader = UtilsIMDB.create_data_loader(selected_train_texts, selected_train_labels, vocab, max_length=200,
                                                    batch_size=64, shuffle=True)
        test_loader = UtilsIMDB.create_data_loader(test_texts, test_labels, vocab, max_length=200, batch_size=64,
                                                   shuffle=False)

        # 创建TextCNN模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IMDBTextCNN(
            vocab_size=len(vocab),
            embed_dim=100,
            filter_sizes=[3, 4, 5],
            num_filters=100,
            num_classes=2,
            dropout=0.5
        ).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                          model_save_path=model_save_path)
        model.save_model(f"{project_root}/data/model/imdb_textcnn_model")

        # 保存词汇表
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    return model


# 给数据集添加噪声
def dataowner_add_noise(dataowners, rate):
    """
    给数据集添加噪声
    :param dataowners:
    :param rate: 加噪的程度，初始程度在0-1之间
    :return:
    """
    # 第一次训练时：添加噪声，以提供不同质量的数据
    for i, do in enumerate(dataowners):
        random_num = random.random() * rate
        noise_type = random.choice(["word_dropout", "word_swap", "word_replace"])
        UtilsIMDB.add_noise(do, noise_type=noise_type, severity=random_num)
        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"DataOwner{i + 1}: noise random: {random_num}, type: {noise_type}")


# ModelOwner发布任务， DataOwner计算数据质量（Dataowner自己计算）
def evaluate_data_quality(dataowners):
    """
    评估DataOwner的数据质量
    :param dataowners: DataOwner对象列表
    :return: 归一化后的数据质量列表
    """
    avg_f_list = []

    # 评价数据质量
    for i, do in enumerate(dataowners):
        quality_scores = UtilsIMDB.evaluate_quality(do, metric="lexical_diversity")

        # 计算平均质量得分
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_f_list.append(avg_quality)
        UtilsIMDB.print_and_log(global_imdb_parent_path, f"DataOwner{i + 1}: 平均质量得分 = {avg_quality:.4f}")

    UtilsIMDB.print_and_log(global_imdb_parent_path, "DataOwners自行评估数据质量：")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
    UtilsIMDB.print_and_log(global_imdb_parent_path,
                            f"归一化后的数据质量列表avg_f_list: {UtilsIMDB.normalize_list(avg_f_list)}")

    return UtilsIMDB.normalize_list(avg_f_list)


# ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
def calculate_optimal_payment_and_data(avg_f_list, last_xn_list):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list: 平均数据质量列表
    :param last_xn_list: 上一轮的数据量列表
    :return: 最优数据量列表，最优支付，ModelOwner效用，DataOwner平均效用
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    eta_opt, x_opt, U_opt = Stackelberg.find_stackelberg_equilibrium(Alpha, np.array(avg_f_list), Lambda, Rho)

    UtilsIMDB.print_and_log(global_imdb_parent_path, "Stackelberg均衡结果：")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"ModelOwner的最优Eta = {eta_opt:.4f}")
    xn_list = []
    for i, xi in enumerate(x_opt):
        UtilsIMDB.print_and_log(global_imdb_parent_path, f"DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
        xn_list.append(xi)
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"每个DataOwner应该贡献数据比例 xn_list = {xn_list}")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"ModelOwner的最大效用 U(Eta) = {U_opt:.4f}")

    # 这里计算 U_Eta 和 U_qn
    U_Eta = U_opt
    U_qn = (eta_opt - Lambda * Rho * (sum(xn_list))) / N

    return UtilsIMDB.compare_elements(xn_list, last_xn_list), eta_opt, U_Eta, U_qn


# DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
def compute_contribution_rates(xn_list, avg_f_list, best_Eta):
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list: 数据量列表
    :param avg_f_list: 平均数据质量列表
    :param best_Eta: 最优总支付
    :return:
    """
    # 计算qn （qn = xn*fn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]

    sum_qn = sum(contributions)

    UtilsIMDB.print_and_log(global_imdb_parent_path, f"ModelOwner的最优总支付：{best_Eta}")
    for i in range(len(xn_list)):
        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"DataOwner{i + 1}的分配到的支付 ： {contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和CPC
def match_data_owners_to_cpc(xn_list, cpcs, dataowners):
    """
    匹配DataOwner和CPC
    :param xn_list: 数据量列表
    :param cpcs: CPC对象列表
    :param dataowners: DataOwner对象列表
    :return: 匹配结果
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)

    preferences = GaleShapley.make_preferences(xn_list, cpcs, Rho, dataowners)

    # 调用Gale-Shapley算法
    matching = GaleShapley.gale_shapley(proposals, preferences)
    UtilsIMDB.print_and_log(global_imdb_parent_path, matching)
    return matching


# DataOwner向CPC提交数据
def submit_data_to_cpc(matching, dataowners, cpcs, xn_list):
    """
    DataOwner按照xn_list中约定的比例向CPC提交数据
    :param matching: 匹配结果
    :param dataowners: DataOwner对象列表
    :param cpcs: CPC对象列表
    :param xn_list: 需要提交的数据的比例
    :return:
    """
    for item in matching.items():
        # 使用正则表达式匹配字符串末尾的数字
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"DataOwner{dataowner_index + 1} 把数据交给 CPC{cpc_index + 1}")

        UtilsIMDB.dataowner_pass_data_to_cpc(dataowners[dataowner_index], cpcs[cpc_index], xn_list[dataowner_index])


# 使用CPC进行模型训练和全局模型的更新
def train_model_with_cpc(matching, cpcs, test_texts, test_labels, literation, avg_f_list, adjustment_literation):
    """
    使用CPC进行模型训练和全局模型的更新
    :param matching: 匹配结果
    :param cpcs: CPC对象列表
    :param test_texts: 测试文本数据
    :param test_labels: 测试标签数据
    :param literation: 训练的轮次
    :param avg_f_list: 平均数据质量列表
    :param adjustment_literation: 要进行调整的轮次
    :return: 更新后的平均数据质量列表
    """

    # 为了找到项目根目录
    def find_project_root(current_dir):
        # 向上逐层查找，直到找到项目根目录
        while not os.path.exists(os.path.join(current_dir, 'README.md')):  # 假设项目根目录包含 README.md 文件
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

    for item in matching.items():
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].textData)}")
        if len(cpcs[cpc_index].textData) == 0:
            UtilsIMDB.print_and_log(global_imdb_parent_path, "数据量为0，跳过此轮调整")
            continue

        # 加载词汇表
        vocab_path = f"{project_root}/data/model/imdb_vocab.pkl"
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # 创建数据加载器
        train_loader = UtilsIMDB.create_data_loader(cpcs[cpc_index].textData, cpcs[cpc_index].labelData,
                                                    vocab, max_length=200, batch_size=64, shuffle=True)
        test_loader = UtilsIMDB.create_data_loader(test_texts, test_labels,
                                                   vocab, max_length=200, batch_size=64, shuffle=False)

        # 创建TextCNN模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IMDBTextCNN(
            vocab_size=len(vocab),
            embed_dim=100,
            filter_sizes=[3, 4, 5],
            num_filters=100,
            num_classes=2,
            dropout=0.5
        ).to(device)

        fine_tune_model(model, train_loader, test_loader, num_epochs=5, device=str(device),
                        lr=1e-5, model_path=f"{project_root}/data/model/imdb_textcnn_model")

    return UtilsIMDB.normalize_list(avg_f_list)


if __name__ == "__main__":
    UtilsIMDB.print_and_log(global_imdb_parent_path,
                            f"**** {global_imdb_parent_path}-MIX 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

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
    for n in [39]:
        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"========================= 客户端数量: {n + 1} =========================")

        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                "---------------------------------- 定义参数值 ----------------------------------")
        Lambda, Rho, Alpha, Epsilon, N, M, SigmaM = define_parameters(Lambda=Lambda, Rho=Rho, Alpha=Alpha,
                                                                      Epsilon=Epsilon, M=n + 1, N=n + 1,
                                                                      SigmaM=[1] * (n + 1))
        UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                "---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, cpcs, test_texts, test_labels = ready_for_task()
        UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

        avg_f_list = []

        # DataOwner自己报数据质量的机会只有一次

        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                f"----- 计算 DataOwner 的数据质量 -----")
        avg_f_list = evaluate_data_quality(dataowners)
        UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

        """下面是三种方案的分别的效用计算"""
        """QD-RDFL"""

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N

        while True:
            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"========================= literation: {literation + 1} =========================")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            xn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list)
            last_xn_list = xn_list

            # 只有在调整轮次之后的轮次才记录
            if literation == adjustment_literation + 1:
                U_Eta_list.append(U_Eta)
                U_qn_list.append(U_qn)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list: {U_Eta_list}")
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list: {U_qn_list}")
                break

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsIMDB.print_and_log(global_imdb_parent_path,
                                        f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_texts, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            literation += 1

        """RANDOM"""
        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                "---------------------------------- RANDOM：再次准备工作 ----------------------------------")
        _, modelowner, _, _, _ = ready_for_task()
        avg_f_list = []
        avg_f_list = evaluate_data_quality(dataowners)
        UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N
        while True:
            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"========================= literation: {literation + 1} =========================")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
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
                    UtilsIMDB.print_and_log(global_imdb_parent_path,
                                            f"random: DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
                    random_xn_list.append(xi)
                # random_xn_list = UtilsIMDB.compare_elements(random_xn_list, [0] * N)
                random_U_Eta = Stackelberg._leader_utility(random_Eta, Alpha, avg_f_list, random_xn_list)
                random_U_qn = (random_Eta - Lambda * Rho * (sum(random_xn_list))) / N

                # 添加进列表
                U_Eta_list_RANDOM.append(random_U_Eta)
                U_qn_list_RANDOM.append(random_U_qn)

                # 更新"上一值"
                last_random_U_Eta = random_U_Eta
                last_random_U_qn = random_U_qn

            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list_RANDOM: {U_Eta_list_RANDOM}")
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list_RANDOM: {U_qn_list_RANDOM}")
                break

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsIMDB.print_and_log(global_imdb_parent_path,
                                        f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_texts, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            literation += 1

        """FIX"""
        UtilsIMDB.print_and_log(global_imdb_parent_path,
                                "---------------------------------- FIX：再次准备工作 ----------------------------------")
        _, modelowner, _, _, _ = ready_for_task()
        avg_f_list = []
        avg_f_list = evaluate_data_quality(dataowners)
        UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_literation = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        last_xn_list = [0] * N
        while True:
            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"========================= literation: {literation + 1} =========================")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
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
                    UtilsIMDB.print_and_log(global_imdb_parent_path,
                                            f"fix: DataOwner{i + 1}的最优x_{i + 1} = {xi:.4f}")
                    fix_xn_list.append(xi)
                # fix_xn_list = UtilsIMDB.compare_elements(fix_xn_list, [0] * N)
                fix_U_Eta = Stackelberg._leader_utility(fix_Eta, Alpha, avg_f_list, fix_xn_list)
                fix_U_qn = (fix_Eta - Lambda * Rho * (sum(fix_xn_list))) / N

                # 添加进列表
                U_Eta_list_FIX.append(fix_U_Eta)
                U_qn_list_FIX.append(fix_U_qn)

                # 更新"上一值"
                last_fix_U_Eta = fix_U_Eta
                last_fix_U_qn = fix_U_qn

            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 提前中止
            if literation > adjustment_literation:
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list_FIX: {U_Eta_list_FIX}")
                UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list_FIX: {U_qn_list_FIX}")
                break

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, best_Eta)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsIMDB.print_and_log(global_imdb_parent_path,
                                        f"----- literation {literation + 1}: 匹配 DataOwner 和 CPC -----")
                matching = match_data_owners_to_cpc(xn_list, cpcs, dataowners)
                UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path,
                                    f"----- literation {literation + 1}: DataOwner 向 CPC 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, cpcs, xn_list)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            UtilsIMDB.print_and_log(global_imdb_parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list = train_model_with_cpc(matching, cpcs, test_texts, test_labels, literation, avg_f_list,
                                              adjustment_literation)
            UtilsIMDB.print_and_log(global_imdb_parent_path, "DONE")

            literation += 1

    UtilsIMDB.print_and_log(global_imdb_parent_path, f"{N}:QD-RDFL的列表：")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list: {U_Eta_list}")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list: {U_qn_list}")

    UtilsIMDB.print_and_log(global_imdb_parent_path, f"{N}:RANDOM的列表：")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list_RANDOM: {U_Eta_list_RANDOM}")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list: {U_qn_list_RANDOM}")

    UtilsIMDB.print_and_log(global_imdb_parent_path, f"{N}:FIX最终的列表：")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_Eta_list_FIX: {U_Eta_list_FIX}")
    UtilsIMDB.print_and_log(global_imdb_parent_path, f"U_qn_list_FIX: {U_qn_list_FIX}")
