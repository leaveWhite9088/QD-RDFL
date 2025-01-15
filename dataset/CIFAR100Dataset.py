import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class CIFAR100Dataset(Dataset):
    def __init__(self, batch_files, transform=None):
        """
        初始化CIFAR-100数据集

        :param batch_files: 包含所有批处理文件路径的列表（例如 ['train', 'test']）
        :param transform: 可选的图像转换（如数据增强、归一化等）
        """
        self.transform = transform
        self.data, self.labels = self._load_batches(batch_files)

    def _load_batches(self, batch_files):
        """
        从批处理文件加载CIFAR-100数据

        :param batch_files: 批处理文件路径列表
        :return: (data, labels) 两个numpy数组
        """
        all_data = []
        all_labels = []
        for batch_file in batch_files:
            if not os.path.isfile(batch_file):
                raise FileNotFoundError(f"Batch file {batch_file} not found.")
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data = batch[b'data']  # shape: (num_samples, 3072)
                labels = batch[b'fine_labels']  # CIFAR-100 使用 'fine_labels'
                all_data.append(data)
                all_labels.extend(labels)
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        all_labels = np.array(all_labels, dtype=np.int64)
        return all_data, all_labels

    def __len__(self):
        """
        返回数据集的大小
        """
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
            # 将 numpy array 转换为 PIL Image 以应用 torchvision.transforms
            image = torch.tensor(image).permute(1, 2, 0)  # 转换为 (H, W, C)
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        return image, label
