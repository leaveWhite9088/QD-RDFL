import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, batch_files, transform=None):
        """
        初始化CIFAR10数据集

        :param batch_files: 包含所有批处理文件路径的列表（例如 ['data_batch_1', ..., 'test_batch']）
        :param transform: 可选的图像转换（如数据增强、归一化等）
        """
        self.transform = transform
        self.data, self.labels = self._load_batches(batch_files)

    def _load_batches(self, batch_files):
        """
        从批处理文件加载CIFAR10数据

        :param batch_files: 批处理文件路径列表
        :return: (data, labels) 两个numpy数组
        """
        all_data = []
        all_labels = []
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                # 'data' 是一个 (10000, 3072) 的数组，每行是一个展平的图像
                data = batch[b'data']
                labels = batch[b'labels']
                all_data.append(data)
                all_labels.extend(labels)
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        all_labels = np.array(all_labels, dtype=np.int64)
        return all_data, all_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取指定索引的图像及其标签

        :param idx: 索引
        :return: (image_tensor, label)
        """
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        return image, label
