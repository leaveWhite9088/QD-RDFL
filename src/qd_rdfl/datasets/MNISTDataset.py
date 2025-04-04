import torch
from torch.utils.data import Dataset
import numpy as np
import struct

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)

    def _load_images(self, path):
        with open(path, 'rb') as f:
            # 读取MNIST图像文件的头部信息
            f.read(16)  # 跳过头部信息
            # 读取图像数据
            images = np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28)  # 每个图像28x28
        return torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0  # 将数据归一化到[0, 1]区间

    def _load_labels(self, path):
        with open(path, 'rb') as f:
            # 读取MNIST标签文件的头部信息
            f.read(8)  # 跳过头部信息
            # 读取标签数据
            labels = np.fromfile(f, dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
