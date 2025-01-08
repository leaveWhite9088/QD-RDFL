import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


class UtilsMNIST:

    # 将MNIST数据集切分成不等的N份，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners(dataowners, X_data, y_data):
        """
        将MNIST数据集切分成不等的N份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如MNIST的图像数据）
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

    # 将MNIST数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners_with_large_gap(dataowners, X_data, y_data):
        """
        将MNIST数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如MNIST的图像数据）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)

        示例：
        X_train = np.random.rand(60000, 784)  # 假设是MNIST的样本数据，每个图像是28x28=784维向量
        y_train = np.random.randint(0, 10, 60000)  # 假设是MNIST的标签，0到9的数字

        # 创建DataOwner对象数组
        dataowners = [DataOwner(Lambda=0.5, Rho=0.1) for _ in range(5)]  # 假设有5个DataOwner

        # 切分数据，差距较大
        Utils.split_data_to_dataowners_with_large_gap(dataowners, X_train, y_train)

        # 检查每个DataOwner持有的数据
        for i, do in enumerate(dataowners):
        print(f"DataOwner {i + 1} holds {len(do.imgData)} samples")

        输出结果：
        Split sizes: [ 3531 28235 17647  7058  3529]
        DataOwner 1 holds 3531 samples
        DataOwner 2 holds 28235 samples
        DataOwner 3 holds 17647 samples
        DataOwner 4 holds 7058 samples
        DataOwner 5 holds 3529 samples
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

    @staticmethod
    def _load_mnist_images(file_path):
        """
        从 .idx3-ubyte 文件中加载 MNIST 图像数据
        :param file_path: 图像文件路径
        :return: numpy 数组 (images)
        """
        with open(file_path, 'rb') as f:
            # 读取文件头信息
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # 检查 magic number 是否匹配
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in file {file_path}")
            # 读取图像数据，并将其转换为 numpy 数组
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)  # 转换为 (num_images, 28, 28)
            return images

    @staticmethod
    def _load_mnist_labels(file_path):
        """
        从 .idx1-ubyte 文件中加载 MNIST 标签数据
        :param file_path: 标签文件路径
        :return: numpy 数组 (labels)
        """
        with open(file_path, 'rb') as f:
            # 读取文件头信息
            magic, num_labels = struct.unpack(">II", f.read(8))
            # 检查 magic number 是否匹配
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in file {file_path}")
            # 读取标签数据，并将其转换为 numpy 数组
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    # 加载 MNIST 数据集（特定文件路径）
    @staticmethod
    def load_mnist_dataset(image_file, label_file):
        """
        加载 MNIST 数据集（特定文件路径）
        :param image_file: 图像文件路径（.idx3-ubyte 文件）
        :param label_file: 标签文件路径（.idx1-ubyte 文件）
        :return: 图像数组和标签数组
        """
        images = UtilsMNIST._load_mnist_images(image_file)
        labels = UtilsMNIST._load_mnist_labels(label_file)
        return images, labels

    # 将图像和标签数组封装成 PyTorch DataLoader
    @staticmethod
    def create_data_loader(images, labels, batch_size=64, shuffle=True):
        """
        将图像和标签数组封装成 PyTorch DataLoader
        :param images: numpy 数组，形状为 (num_samples, 28, 28)
        :param labels: numpy 数组，形状为 (num_samples,)
        :param batch_size: 每个批次的样本数
        :param shuffle: 是否打乱数据
        :return: PyTorch DataLoader

        示例：
        # 定义文件路径
        train_images_path = "./dataset/MNIST/train-images.idx3-ubyte"
        train_labels_path = "./dataset/MNIST/train-labels.idx1-ubyte"
        test_images_path = "./dataset/MNIST/t10k-images.idx3-ubyte"
        test_labels_path = "./dataset/MNIST/t10k-labels.idx1-ubyte"

        # 加载训练数据和测试数据
        train_images, train_labels = Utils.load_mnist_dataset(train_images_path, train_labels_path)
        test_images, test_labels = Utils.load_mnist_dataset(test_images_path, test_labels_path)

        # 创建 DataLoader
        train_loader = create_data_loader(train_images, train_labels, batch_size=64, shuffle=True)
        test_loader = create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)
        """
        # 转换图像为 float32 并归一化到 [0.0, 1.0]
        images = images.astype(np.float32)
        # 增加通道维度，变为 (num_samples, 1, 28, 28)
        images = np.expand_dims(images, axis=1)
        # 转换为 PyTorch 张量
        images_tensor = torch.tensor(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # 创建 TensorDataset
        dataset = TensorDataset(images_tensor, labels_tensor)

        # 创建 DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    # 从两个长度相同的数组中随机选取相同位置的元素，形成两个新的数组。
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
            if proportion < 0:
                proportion = 0
            if proportion > 1:
                proportion = 1

        # 计算需要选取的元素数量
        num_samples = int(len(arr1) * proportion)

        # 随机生成索引
        indices = np.random.choice(len(arr1), num_samples, replace=False)

        # 使用随机索引选取数据
        sampled_arr1 = arr1[indices]
        sampled_arr2 = arr2[indices]

        return sampled_arr1, sampled_arr2

    # 给 imgData 中的每个图像添加噪声，使图像质量变差
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
                noisy_data.append(UtilsMNIST._add_gaussian_noise(img, severity))
            elif noise_type == "salt_and_pepper":
                noisy_data.append(UtilsMNIST._add_salt_and_pepper_noise(img, severity))
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
        :param img: 单张图像数据 (numpy array)
        :param severity: 高斯噪声的标准差比例（0-1）
        :return: 添加噪声后的图像
        """
        noise = np.random.normal(0, severity, img.shape)  # 生成高斯噪声
        noisy_img = img * 1.0 + noise  # 添加噪声
        noisy_img = np.clip(noisy_img, 0, 1)  # 保证像素值在 [0, 1] 范围内
        return noisy_img

    @staticmethod
    def _add_salt_and_pepper_noise(img, severity):
        """
        内部方法：给每张图像添加椒盐噪声
        :param img: 图像数据，形状为 (num_samples, 28, 28)
        :param severity: 噪声强度（0-1），表示椒盐噪声的比例
        :return: 添加噪声后的图像
        """
        noisy_img = img.copy()
        N, H, W = noisy_img.shape
        num_salt = int(np.ceil(severity * H * W * 0.5))
        num_pepper = int(np.ceil(severity * H * W * 0.5))

        for i in range(N):
            # 添加盐噪声
            coords = [np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt)]
            noisy_img[i, coords[0], coords[1]] = 1

            # 添加椒噪声
            coords = [np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper)]
            noisy_img[i, coords[0], coords[1]] = 0

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
                quality_scores.append(UtilsMNIST._calculate_mse(original, noisy))
            elif metric == "snr":
                quality_scores.append(UtilsMNIST._calculate_snr(original, noisy))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return quality_scores

    @staticmethod
    def _calculate_mse(original, noisy):
        """
        计算单张图像的均方误差 (MSE)
        :param original: 原始图像
        :param noisy: 噪声图像
        :return: 均方误差
        """
        mse = np.mean((original - noisy) ** 2)
        return mse

    @staticmethod
    def _calculate_snr(original, noisy):
        """
        计算单张图像的信噪比 (SNR)
        :param original: 原始图像
        :param noisy: 噪声图像
        :return: 信噪比 (dB)
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - noisy) ** 2)
        if noise_power == 0:
            return float('inf')  # 无噪声时返回无限大
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    # dataowner把数据传给cpc
    @staticmethod
    def dataowner_pass_data_to_cpc(dataowner, cpc, proportion):
        """
        dataowner把数据传给cpc
        :param dataowner:
        :param cpc:
        :param proportion:比例
        :return:
        """
        cpc.imgData, cpc.labelData = UtilsMNIST.sample_arrays(np.array(dataowner.imgData),
                                                              np.array(dataowner.labelData), proportion)

    # 定义一个函数，用于同时打印到控制台和文件
    @staticmethod
    def print_and_log(message):
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(current_file_path)

        # 获取项目根目录（假设项目根目录是当前文件所在目录的父目录）
        project_root = os.path.dirname(current_dir)

        # 构建日志文件的完整路径
        log_file_path = os.path.join(project_root, 'data', 'log', 'log-MNIST.txt')

        # 打开一个文件用于追加写入
        with open(log_file_path, 'a') as f:
            # 将message转换为字符串
            message_str = str(message)

            # 打印到控制台
            print(message_str)

            # 写入到文件
            f.write(message_str + '\n')

    # 归一化列表
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

        normalized_lst = [0.5 + 0.5 * ((x - min_val) / (max_val - min_val)) for x in lst]
        return normalized_lst

    # 取两个list中较大的元素
    @staticmethod
    def compare_elements(list1, list2):
        # 使用列表推导式和zip函数逐个比较元素
        comparison_results = [x if x > y else y for x, y in zip(list1, list2)]
        return comparison_results
