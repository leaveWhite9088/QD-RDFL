import torch
from torch.utils.data import Dataset
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle

class IMDBDataset(Dataset):
    def __init__(self, data_dir, mode='train', max_length=200, vocab=None, vocab_size=10000):
        """
        初始化IMDB数据集
        :param data_dir: 数据集所在目录
        :param mode: 'train' 或 'test'
        :param max_length: 每条评论保留的最大单词数
        :param vocab: 词汇表，如果为None则会从训练数据创建
        :param vocab_size: 词汇表大小
        """
        super(IMDBDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.max_length = max_length
        
        # 获取文件路径
        pos_dir = os.path.join(data_dir, mode, 'pos')
        neg_dir = os.path.join(data_dir, mode, 'neg')
        
        # 读取文本和标签
        self.texts = []
        self.labels = []
        
        # 读取正面评论
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    self.texts.append(f.read())
                    self.labels.append(1)  # 1表示正面评论
        
        # 读取负面评论
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    self.texts.append(f.read())
                    self.labels.append(0)  # 0表示负面评论
        
        # 创建或加载词汇表
        if vocab is None and mode == 'train':
            print("创建词汇表...")
            self.vocab = self._create_vocab(vocab_size)
        else:
            self.vocab = vocab
        
        # 将文本转换为数字序列
        self.sequences = [self._text_to_sequence(text) for text in self.texts]

    def _create_vocab(self, vocab_size):
        """
        从训练数据创建词汇表
        :param vocab_size: 词汇表大小
        :return: 词汇表（单词到索引的映射）
        """
        # 显式设置nltk数据路径
        nltk.data.path.append("D:/base/code/IDE/Anaconda/anaconda3/envs/py308base/nltk_data")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        word_counts = Counter()
        for text in self.texts:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
            words = word_tokenize(text)
            word_counts.update(words)
        
        # 创建词汇表（保留最常见的单词）
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common(vocab_size - 2):  # -2 是因为有PAD和UNK
            vocab[word] = len(vocab)
        
        return vocab

    def _text_to_sequence(self, text):
        """
        将文本转换为数字序列
        :param text: 文本字符串
        :return: 数字序列
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
        words = word_tokenize(text)
        
        sequence = []
        for word in words[:self.max_length]:
            if word in self.vocab:
                sequence.append(self.vocab[word])
            else:
                sequence.append(self.vocab['<UNK>'])
        
        # 填充或截断到指定长度
        if len(sequence) < self.max_length:
            sequence += [self.vocab['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long) 