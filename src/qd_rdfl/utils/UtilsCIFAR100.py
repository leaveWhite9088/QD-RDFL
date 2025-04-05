import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle

class UtilsCIFAR100:
    @staticmethod
    def split_data_to_dataowners(dataowners, X_data, y_data):
        """
        将CIFAR-100数据集切分成不等的N份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR-100的图像数据，形状为 (num_samples, 3, 32, 32)）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 生成一个不等的切分比例（可以根据需要修改）
        # 例如我们将每个DataOwner分配不同数量的样本，这里以随机切分为例
        split_sizes = np.random.multinomial(total_samples, [1 / N] * N)

        start_idx = 0
        for i, do in enumerate(dataowners):
            end_idx = start_idx + split_sizes[i]
            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]
            start_idx = end_idx

    @staticmethod
    def split_data_to_dataowners_with_large_gap(dataowners, X_data, y_data):
        """
        将CIFAR-100数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR-100的图像数据，形状为 (num_samples, 3, 32, 32)）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 创建一个不均匀的权重分布，差距大
        # 比如用一个几何分布生成差距大的权重，然后归一化
        raw_weights = np.random.geometric(p=0.2, size=N)  # 几何分布会生成较大的差距
        weights = raw_weights / raw_weights.sum()  # 归一化权重，确保总和为1

        # 根据权重生成切分比例
        split_sizes = (weights * total_samples).astype(int)

        # 修正分配：由于整数化可能导致分配的样本数不完全等于total_samples
        diff = total_samples - split_sizes.sum()
        split_sizes[0] += diff  # 将差值调整到第一个DataOwner

        start_idx = 0
        for i, do in enumerate(dataowners):
            end_idx = start_idx + split_sizes[i]
            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]
            start_idx = end_idx

        # 输出权重分布和切分情况
        print("Weights:", weights)
        print("Split sizes:", split_sizes)

    # 将CIFAR10数据集切分成N等份，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners_evenly(dataowners, X_data, y_data):
        """
        将CIFAR10数据集切分成N等份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR10的图像数据）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 计算每个DataOwner应分配的样本数量
        samples_per_owner = total_samples // N
        remainder = total_samples % N  # 处理不能整除的情况

        start_idx = 0
        for i, do in enumerate(dataowners):
            # 如果有余数，将余数分配到前面的DataOwner
            extra_samples = 1 if i < remainder else 0
            end_idx = start_idx + samples_per_owner + extra_samples

            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]

            start_idx = end_idx

        # 输出分配情况
        for i, do in enumerate(dataowners):
            print(f"DataOwner {i + 1} holds {len(do.imgData)} samples")

    @staticmethod
    def _load_batch(batch_file):
        """
        从批处理文件加载CIFAR-100数据
        :param batch_file: 批处理文件路径
        :return: (data, labels) 两个numpy数组
        """
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']  # shape: (num_samples, 3072)
            labels = batch[b'fine_labels']  # CIFAR-100 使用 'fine_labels'
            data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化到 [0, 1]
            labels = np.array(labels, dtype=np.int64)
        return data, labels

    @staticmethod
    def load_cifar100_dataset(data_dir):
        """
        加载CIFAR-100数据集
        :param data_dir: CIFAR-100数据集所在的目录，包含 'train', 'test', 'meta' 文件
        :return: (train_data, train_labels, test_data, test_labels)
        """
        # 加载训练数据
        train_batch_file = os.path.join(data_dir, 'train')
        train_data, train_labels = UtilsCIFAR100._load_batch(train_batch_file)

        # 加载测试数据
        test_batch_file = os.path.join(data_dir, 'test')
        test_data, test_labels = UtilsCIFAR100._load_batch(test_batch_file)

        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_meta(data_dir):
        """
        加载CIFAR-100的元数据（如标签名称）
        :param data_dir: CIFAR-100数据集所在的目录，包含 'meta' 文件
        :return: fine_label_names, coarse_label_names
        """
        meta_file = os.path.join(data_dir, 'meta')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            fine_label_names = meta[b'fine_label_names']
            coarse_label_names = meta[b'coarse_label_names']
        # 将标签名称从字节转换为字符串
        fine_label_names = [label.decode('utf-8') for label in fine_label_names]
        coarse_label_names = [label.decode('utf-8') for label in coarse_label_names]
        return fine_label_names, coarse_label_names

    @staticmethod
    def create_data_loader(images, labels, batch_size=64, shuffle=True, num_workers=4):
        """
        创建DataLoader对象
        :param images: 图像数据，形状为 (num_samples, 3, 32, 32)
        :param labels: 标签数据，形状为 (num_samples,)
        :param batch_size: 每个batch的大小
        :param shuffle: 是否打乱数据
        :param num_workers: 数据加载时使用的子进程数
        :return: DataLoader对象
        """
        tensor_x = torch.tensor(images, dtype=torch.float32)
        tensor_y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def sample_arrays(arr1, arr2, proportion):
        """
        从两个长度相同的数组中随机选取相同位置的元素，形成两个新的数组。

        参数:
        arr1 (np.array): 第一个数组。
        arr2 (np.array): 第二个数组。
        proportion (float): 选取的比例，范围在0到1之间。

        返回:
        np.array: 从arr1中选取的新数组。
        np.array: 从arr2中选取的新数组。
        """
        if len(arr1) != len(arr2):
            raise ValueError("两个数组的长度必须相同")
        if not (0 <= proportion <= 1):
            print("比例必须在0到1之间，已自动调整")
            proportion = np.clip(proportion, 0, 1)

        # 计算需要选取的元素数量
        num_samples = int(len(arr1) * proportion)

        # 随机生成索引
        indices = np.random.choice(len(arr1), num_samples, replace=False)

        # 使用随机索引选取数据
        sampled_arr1 = arr1[indices]
        sampled_arr2 = arr2[indices]

        return sampled_arr1, sampled_arr2

    @staticmethod
    def add_noise(dataowner, noise_type="gaussian", severity=0.1):
        """
        给 imgData 中的每个图像添加噪声，使图像质量变差
        :param noise_type: 噪声类型，"gaussian" 或 "salt_and_pepper"
        :param severity: 噪声的严重程度（0-1）
        """
        # 原数据进行加噪处理
        noisy_data = []
        for img in dataowner.imgData:
            if noise_type == "gaussian":
                noisy_data.append(UtilsCIFAR100._add_gaussian_noise(img, severity))
            elif noise_type == "salt_and_pepper":
                noisy_data.append(UtilsCIFAR100._add_salt_and_pepper_noise(img, severity))
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
        dataowner.imgData = noisy_data

        # 原保留数据进行归一化
        temp_data = []
        for img in dataowner.originalData:
            temp_img = np.clip(img, 0, 1)  # 保证像素值在 [0, 1] 范围内
            temp_data.append(temp_img)
        dataowner.originalData = temp_data

    @staticmethod
    def _add_gaussian_noise(img, severity):
        """
        内部方法：给单张图像添加高斯噪声
        :param img: 单张图像数据 (numpy array)，形状为 (3, 32, 32)
        :param severity: 高斯噪声的标准差比例（0-1）
        :return: 添加噪声后的图像
        """
        noise = np.random.normal(0, severity, img.shape)  # 生成高斯噪声
        noisy_img = img + noise  # 添加噪声
        noisy_img = np.clip(noisy_img, 0, 1)  # 保证像素值在 [0, 1] 范围内
        return noisy_img

    @staticmethod
    def _add_salt_and_pepper_noise(img, severity):
        """
        内部方法：给每张图像添加椒盐噪声
        :param img: 图像数据，形状为 (3, 32, 32)
        :param severity: 噪声强度（0-1），表示椒盐噪声的比例
        :return: 添加噪声后的图像
        """
        noisy_img = img.copy()
        C, H, W = noisy_img.shape
        num_salt = int(np.ceil(severity * H * W * 0.5))
        num_pepper = int(np.ceil(severity * H * W * 0.5))

        for c in range(C):
            # 添加盐噪声
            coords = [np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt)]
            noisy_img[c, coords[0], coords[1]] = 1

            # 添加椒噪声
            coords = [np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper)]
            noisy_img[c, coords[0], coords[1]] = 0

        return noisy_img

    @staticmethod
    def evaluate_quality(dataowner, metric="mse"):
        """
        评价 imgData 的质量
        :param metric: 评价指标类型，支持 "mse" 或 "snr"
        :return: 数据质量得分列表（每张图像的质量得分）
        """
        if len(dataowner.originalData) != len(dataowner.imgData):
            raise ValueError("originalData and imgData must have the same length.")

        quality_scores = []
        for original, noisy in zip(dataowner.originalData, dataowner.imgData):
            if metric == "mse":
                quality_scores.append(UtilsCIFAR100._calculate_mse(original, noisy))
            elif metric == "snr":
                quality_scores.append(UtilsCIFAR100._calculate_snr(original, noisy))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return quality_scores

    @staticmethod
    def _calculate_mse(original, noisy):
        """
        计算单张图像的均方误差 (MSE)
        :param original: 原始图像，形状为 (3, 32, 32)
        :param noisy: 噪声图像，形状为 (3, 32, 32)
        :return: 均方误差
        """
        mse = np.mean((original - noisy) ** 2)
        return mse

    @staticmethod
    def _calculate_snr(original, noisy):
        """
        计算单张图像的信噪比 (SNR)
        :param original: 原始图像，形状为 (3, 32, 32)
        :param noisy: 噪声图像，形状为 (3, 32, 32)
        :return: 信噪比 (dB)
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - noisy) ** 2)
        if noise_power == 0:
            return float('inf')  # 无噪声时返回无限大
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def dataowner_pass_data_to_cpc(dataowner, cpc, proportion):
        """
        dataowner把数据传给cpc
        :param dataowner:
        :param cpc:
        :param proportion:比例
        :return:
        """
        cpc.imgData, cpc.labelData = UtilsCIFAR100.sample_arrays(np.array(dataowner.imgData),
                                                                   np.array(dataowner.labelData), proportion)

    @staticmethod
    def print_and_log(cifar_parent_path, message):
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(current_file_path)

        # 查找项目根目录（假设项目根目录包含 README.md 文件）
        def find_project_root(current_dir):
            # 向上逐层查找，直到找到项目根目录
            while not os.path.exists(os.path.join(current_dir, 'README.md')):
                current_dir = os.path.dirname(current_dir)
                # 防止在 Unix/Linux 系统中向上查找过多
                if current_dir == '/' or (os.name == 'nt' and current_dir == os.path.splitdrive(current_dir)[0] + '\\'):
                    return None
            return current_dir

        # 查找项目根目录
        project_root = find_project_root(current_dir)

        if project_root is None:
            raise FileNotFoundError("未找到项目根目录，请确保项目根目录包含 README.md 文件")

        # 构建日志文件的完整路径
        log_file_path = os.path.join(project_root, 'data', 'log', cifar_parent_path, f'{cifar_parent_path}-CIFAR100.txt')

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # 打开一个文件用于追加写入
        with open(log_file_path, 'a') as f:
            # 将message转换为字符串
            message_str = str(message)

            # 打印到控制台
            print(message_str)

            # 写入到文件
            f.write(message_str + '\n')

    @staticmethod
    def normalize_list(lst):
        """
        因为等于0会出问题，这里将原始的归一化结果乘以0.5，然后加上0.5，从而将结果从0到1映射到0.5到1。
        :param lst:
        :return:
        """
        if not lst:  # 如果列表为空，返回空列表
            return []

        min_val = min(lst)
        max_val = max(lst)

        if max_val == min_val:  # 如果所有元素都相同，直接返回全1列表
            return [1] * len(lst)

        normalized_lst = [0.9 + 0.1 * ((x - min_val) / (max_val - min_val)) for x in lst]
        return normalized_lst

    @staticmethod
    def compare_elements(list1, list2):
        """
        取两个list中较大的元素
        """
        # 使用列表推导式和zip函数逐个比较元素
        comparison_results = [x if x > y else y for x, y in zip(list1, list2)]
        return comparison_results
