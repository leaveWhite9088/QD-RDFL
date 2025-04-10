import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import re
import nltk
from nltk.tokenize import word_tokenize
import random
import pickle

class UtilsIMDB:

    # 将IMDB数据集切分成不等的N份，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners(dataowners, texts, labels):
        """
        将IMDB数据集切分成不等的N份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param texts: 数据集的特征（文本数据）
        :param labels: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(texts)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        texts_shuffled = [texts[i] for i in permutation]
        labels_shuffled = [labels[i] for i in permutation]

        # 生成一个不等的切分比例
        split_sizes = np.random.multinomial(total_samples, [1 / N] * N)

        start_idx = 0
        for i, do in enumerate(dataowners):
            end_idx = start_idx + split_sizes[i]
            do.textData = texts_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = texts_shuffled[start_idx:end_idx].copy()
            do.labelData = labels_shuffled[start_idx:end_idx]
            start_idx = end_idx

    # 将IMDB数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners_with_large_gap(dataowners, texts, labels):
        """
        将IMDB数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param texts: 数据集的特征（文本数据）
        :param labels: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(texts)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        texts_shuffled = [texts[i] for i in permutation]
        labels_shuffled = [labels[i] for i in permutation]

        # 创建一个不均匀的权重分布，差距大
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
            do.textData = texts_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = texts_shuffled[start_idx:end_idx].copy()
            do.labelData = labels_shuffled[start_idx:end_idx]
            start_idx = end_idx

        # 输出权重分布和切分情况
        print("Weights:", weights)
        print("Split sizes:", split_sizes)

    # 将IMDB数据集切分成N等份，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners_evenly(dataowners, texts, labels):
        """
        将IMDB数据集切分成N等份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param texts: 数据集的特征（文本数据）
        :param labels: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(texts)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        texts_shuffled = [texts[i] for i in permutation]
        labels_shuffled = [labels[i] for i in permutation]

        # 计算每个DataOwner应分配的样本数量
        samples_per_owner = total_samples // N
        remainder = total_samples % N  # 处理不能整除的情况

        start_idx = 0
        for i, do in enumerate(dataowners):
            # 如果有余数，将余数分配到前面的DataOwner
            extra_samples = 1 if i < remainder else 0
            end_idx = start_idx + samples_per_owner + extra_samples

            do.textData = texts_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = texts_shuffled[start_idx:end_idx].copy()
            do.labelData = labels_shuffled[start_idx:end_idx]

            start_idx = end_idx

        # 输出分配情况
        for i, do in enumerate(dataowners):
            print(f"DataOwner {i + 1} holds {len(do.textData)} samples")

    # 加载IMDB数据集
    @staticmethod
    def load_imdb_dataset(data_dir, mode='train', max_length=200, vocab=None, vocab_size=10000):
        """
        加载IMDB数据集
        :param data_dir: 数据集所在目录
        :param mode: 'train' 或 'test'
        :param max_length: 每条评论保留的最大单词数
        :param vocab: 词汇表，如果为None则会从训练数据创建
        :param vocab_size: 词汇表大小
        :return: 文本列表和标签列表
        """
        # 获取文件路径
        pos_dir = os.path.join(data_dir, mode, 'pos')
        neg_dir = os.path.join(data_dir, mode, 'neg')
        
        # 读取文本和标签
        texts = []
        labels = []
        
        # 读取正面评论
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1)  # 1表示正面评论
        
        # 读取负面评论
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(0)  # 0表示负面评论

        return texts, labels

    # 创建词汇表
    @staticmethod
    def create_vocab(texts, vocab_size=10000):
        """
        从训练数据创建词汇表
        :param texts: 文本列表
        :param vocab_size: 词汇表大小
        :return: 词汇表（单词到索引的映射）
        """
        # 显式设置nltk数据路径
        nltk.data.path.append("/root/miniconda3/envs/py38/nltk_data")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
            words = word_tokenize(text)
            word_counts.update(words)
        
        # 创建词汇表（保留最常见的单词）
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common(vocab_size - 2):  # -2 是因为有PAD和UNK
            vocab[word] = len(vocab)
        
        return vocab

    # 将文本转换为数字序列
    @staticmethod
    def text_to_sequence(text, vocab, max_length=200):
        """
        将文本转换为数字序列
        :param text: 文本字符串
        :param vocab: 词汇表
        :param max_length: 序列最大长度
        :return: 数字序列
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
        words = word_tokenize(text)
        
        sequence = []
        for word in words[:max_length]:
            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])
        
        # 填充或截断到指定长度
        if len(sequence) < max_length:
            sequence += [vocab['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence

    # 将文本数据和标签转换为DataLoader
    @staticmethod
    def create_data_loader(texts, labels, vocab, max_length=200, batch_size=64, shuffle=True):
        """
        将文本数据和标签转换为PyTorch DataLoader
        :param texts: 文本列表
        :param labels: 标签列表
        :param vocab: 词汇表
        :param max_length: 序列最大长度
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据
        :return: PyTorch DataLoader
        """
        # 将文本转换为序列
        sequences = [UtilsIMDB.text_to_sequence(text, vocab, max_length) for text in texts]
        
        # 转换为PyTorch张量
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 创建TensorDataset
        dataset = TensorDataset(sequences_tensor, labels_tensor)
        
        # 创建DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return data_loader

    # 从两个长度相同的数组中随机选取相同位置的元素，形成两个新的数组
    @staticmethod
    def sample_arrays(arr1, arr2, proportion):
        """
        从两个长度相同的数组中随机选取相同位置的元素，形成两个新的数组。
        
        参数:
        arr1 (list/np.array): 第一个数组。
        arr2 (list/np.array): 第二个数组。
        proportion (float): 选取的比例，范围在0到1之间。
        
        返回:
        list/np.array: 从arr1中选取的新数组。
        list/np.array: 从arr2中选取的新数组。
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
        if isinstance(arr1, np.ndarray):
            sampled_arr1 = arr1[indices]
            sampled_arr2 = arr2[indices]
        else:
            sampled_arr1 = [arr1[i] for i in indices]
            sampled_arr2 = [arr2[i] for i in indices]
        
        return sampled_arr1, sampled_arr2

    # 给文本数据添加噪声，使文本质量变差
    @staticmethod
    def add_noise(dataowner, noise_type="word_dropout", severity=0.1):
        """
        给textData中的每个文本添加噪声，使文本质量变差
        :param dataowner: 数据拥有者
        :param noise_type: 噪声类型，"word_dropout"、"word_swap"或"word_replace"
        :param severity: 噪声的严重程度（0-1）
        """
        # 原数据进行加噪处理
        noisy_data = []
        for text in dataowner.textData:
            if noise_type == "word_dropout":
                noisy_data.append(UtilsIMDB._add_word_dropout(text, severity))
            elif noise_type == "word_swap":
                noisy_data.append(UtilsIMDB._add_word_swap(text, severity))
            elif noise_type == "word_replace":
                noisy_data.append(UtilsIMDB._add_word_replace(text, severity))
            else:
                raise ValueError(f"不支持的噪声类型: {noise_type}")
        dataowner.textData = noisy_data

    @staticmethod
    def _add_word_dropout(text, severity):
        """
        内部方法：通过随机删除单词来添加噪声
        :param text: 文本字符串
        :param severity: 删除单词的比例（0-1）
        :return: 添加噪声后的文本
        """
        words = text.split()
        if not words:
            return text
        
        # 确定要删除的单词数量
        num_words_to_drop = int(len(words) * severity)
        
        # 随机选择要删除的单词索引
        indices_to_drop = random.sample(range(len(words)), num_words_to_drop)
        
        # 保留未被删除的单词
        noisy_words = [word for i, word in enumerate(words) if i not in indices_to_drop]
        
        return ' '.join(noisy_words)

    @staticmethod
    def _add_word_swap(text, severity):
        """
        内部方法：通过随机交换相邻单词来添加噪声
        :param text: 文本字符串
        :param severity: 交换单词对的比例（0-1）
        :return: 添加噪声后的文本
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        # 确定要交换的单词对数量
        num_pairs_to_swap = int((len(words) - 1) * severity)
        
        # 随机选择要交换的单词对的起始索引
        indices_to_swap = random.sample(range(len(words) - 1), min(num_pairs_to_swap, len(words) - 1))
        
        # 交换选定的单词对
        for index in indices_to_swap:
            words[index], words[index + 1] = words[index + 1], words[index]
        
        return ' '.join(words)

    @staticmethod
    def _add_word_replace(text, severity):
        """
        内部方法：通过随机替换单词为"UNK"来添加噪声
        :param text: 文本字符串
        :param severity: 替换单词的比例（0-1）
        :return: 添加噪声后的文本
        """
        words = text.split()
        if not words:
            return text
        
        # 确定要替换的单词数量
        num_words_to_replace = int(len(words) * severity)
        
        # 随机选择要替换的单词索引
        indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
        
        # 替换选定的单词
        for index in indices_to_replace:
            words[index] = "UNK"
        
        return ' '.join(words)

    # 评价文本质量
    @staticmethod
    def evaluate_quality(dataowner, metric="lexical_diversity"):
        """
        评价textData的质量
        :param dataowner: 数据拥有者
        :param metric: 评价指标类型，支持"lexical_diversity"或"text_length"
        :return: 数据质量得分列表（每个文本的质量得分）
        """
        if len(dataowner.originalData) != len(dataowner.textData):
            raise ValueError("originalData和textData必须具有相同的长度")
        
        quality_scores = []
        for original, noisy in zip(dataowner.originalData, dataowner.textData):
            if metric == "lexical_diversity":
                quality_scores.append(UtilsIMDB._calculate_lexical_diversity(noisy))
            elif metric == "text_length":
                quality_scores.append(UtilsIMDB._calculate_text_length_ratio(original, noisy))
            else:
                raise ValueError(f"不支持的指标: {metric}")
        return quality_scores

    @staticmethod
    def _calculate_lexical_diversity(text):
        """
        计算文本的词汇多样性（不同单词数量除以总单词数量）
        :param text: 文本字符串
        :return: 词汇多样性得分
        """
        words = text.lower().split()
        if not words:
            return 0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        return diversity

    @staticmethod
    def _calculate_text_length_ratio(original, noisy):
        """
        计算噪声文本与原始文本的长度比率
        :param original: 原始文本
        :param noisy: 噪声文本
        :return: 长度比率
        """
        original_length = len(original.split())
        noisy_length = len(noisy.split())
        
        if original_length == 0:
            return 0
        
        ratio = noisy_length / original_length
        return ratio

    # dataowner把数据传给cpc
    @staticmethod
    def dataowner_pass_data_to_cpc(dataowner, cpc, proportion):
        """
        dataowner把数据传给cpc
        :param dataowner: 数据拥有者
        :param cpc: 中央计算提供方
        :param proportion: 比例
        :return: None
        """
        cpc.textData, cpc.labelData = UtilsIMDB.sample_arrays(dataowner.textData, dataowner.labelData, proportion)

    # 定义一个函数，用于同时打印到控制台和文件
    @staticmethod
    def print_and_log(imdb_parent_path, message):
        """
        同时打印到控制台和文件
        :param imdb_parent_path: 父路径名称，用于构建日志文件路径
        :param message: 要打印和记录的消息
        """
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
        log_file_path = os.path.join(project_root, 'data', 'log', imdb_parent_path, f'{imdb_parent_path}-IMDB.txt')

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # 打开一个文件用于追加写入
        with open(log_file_path, 'a', encoding='utf-8') as f:
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
        归一化列表值到0.5到1之间
        :param lst: 输入列表
        :return: 归一化后的列表
        """
        if not lst:  # 如果列表为空，返回空列表
            return []

        min_val = min(lst)
        max_val = max(lst)

        if max_val == min_val:  # 如果所有元素都相同，直接返回全1列表
            return [1] * len(lst)

        normalized_lst = [0.9 + 0.1 * ((x - min_val) / (max_val - min_val)) for x in lst]
        return normalized_lst

    # 取两个list中较大的元素
    @staticmethod
    def compare_elements(list1, list2):
        """
        取两个list中较大的元素
        :param list1: 第一个列表
        :param list2: 第二个列表
        :return: 取较大元素后的新列表
        """
        # 使用列表推导式和zip函数逐个比较元素
        comparison_results = [x if x > y else y for x, y in zip(list1, list2)]
        return comparison_results 