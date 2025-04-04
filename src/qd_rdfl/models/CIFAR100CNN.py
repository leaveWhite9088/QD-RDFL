import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from src.qd_rdfl.datasets.CIFAR100Dataset import CIFAR100Dataset
from src.qd_rdfl.utils.UtilsCIFAR100 import UtilsCIFAR100
from src.qd_rdfl.global_variable import global_cifar100_parent_path

class CIFAR100CNN(nn.Module):
    def __init__(self, num_classes=100):
        """
        初始化CNN模型
        :param num_classes: 输出类别数量
        """
        super(CIFAR100CNN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出通道数调整为128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 保证池化后尺寸减半

        # 定义全连接层
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 输入特征是 16x16x128
        self.fc2 = nn.Linear(256, num_classes)  # 输出类别数量

        # 添加一个额外的卷积层和池化层以增强模型复杂度
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 第二次池化
        self.fc3 = nn.Linear(256 * 8 * 8, 512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 模型输出
        """
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = self.pool(x)            # 第一次池化

        x = F.relu(self.conv3(x))  # 第三层卷积 + ReLU
        x = self.pool2(x)           # 第二次池化

        x = x.view(-1, 256 * 8 * 8) # 展平，这里使用正确的展平维度
        x = F.relu(self.fc3(x))     # 全连接层3 + ReLU
        x = self.fc4(x)             # 全连接层4（输出层）
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs=20, device='cpu',
                   model_save_path=None):
        """
        训练模型并保存最终模型
        :param train_loader: 训练数据加载器
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param num_epochs: 训练轮数
        :param device: 计算设备（'cpu' 或 'cuda'）
        :param model_save_path: 模型保存路径
        """
        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"Epoch {epoch + 1}/{num_epochs} 开始...")  # 打印每个epoch的开始

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累加损失
                running_loss += loss.item()

                # 每 100 个 batch 输出一次损失
                if batch_idx % 100 == 0:
                    UtilsCIFAR100.print_and_log(global_cifar100_parent_path,
                        f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 每个 epoch 结束时输出平均损失
            avg_loss = running_loss / len(train_loader)
            UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"Epoch {epoch + 1} 完成。平均损失: {avg_loss:.4f}")

        if model_save_path is not None:
            # 保存最终模型
            self.save_model(model_save_path)

    def evaluate(self, test_loader, device='cpu'):
        """
        评估模型
        :param test_loader: 数据加载器（测试集）
        :param device: 计算设备（'cpu' 或 'cuda'）
        :return: 模型在测试集上的损失和准确率
        """
        self.to(device)
        self.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
        return avg_loss, accuracy

    def save_model(self, file_path):
        """
        保存模型
        :param file_path: 保存模型的文件路径
        """
        torch.save(self.state_dict(), file_path)
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"模型已保存至 {file_path}")

    def load_model(self, file_path):
        """
        加载模型
        :param file_path: 模型文件路径
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"模型已从 {file_path} 加载")

# 使用CIFAR100数据集，训练cnn
def train_cifar100_cnn(data_dir, model_save_path="../data/model/cifar100_cnn_model"):
    """
    使用CIFAR100数据集，训练cnn
    :param data_dir: CIFAR100数据集所在的目录，包含批处理文件
    :param model_save_path: 保存的路径
    :return:
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    # 加载训练和测试数据
    train_data, train_labels, test_data, test_labels = UtilsCIFAR100.load_cifar100_dataset(data_dir)

    # 创建训练和测试 DataLoader
    train_loader = UtilsCIFAR100.create_data_loader(train_data, train_labels, batch_size=128, shuffle=True)
    test_loader = UtilsCIFAR100.create_data_loader(test_data, test_labels, batch_size=128, shuffle=False)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR100CNN(num_classes=100).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train_model(train_loader, criterion, optimizer, num_epochs=20, device=str(device),
                     model_save_path=model_save_path)

    # 评估模型
    model.evaluate(test_loader, device=str(device))


# 微调整个网络
def fine_tune_model(model, train_loader, test_loader, num_epochs=5, device='cpu', lr=1e-5, model_path=None):
    """
    微调整个网络，但不更新模型，用于计算数据质量
    :param model: 已训练的CNN模型 (CIFAR100CNN)
    :param train_loader: 训练数据加载器，其中包含数据
    :param test_loader: 测试数据加载器
    :param num_epochs: 训练的轮数
    :param device: 计算设备 ('cpu' 或 'cuda')
    :param lr: 微调时的学习率
    :param model_path: 加载模型的路径
    :return: 微调后的单位数据 loss 差
    """

    # 将模型移动到指定设备并转换为Float类型
    model.to(device).float()

    # 加载预训练模型权重（如果有保存的模型）
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)  # 加载先前保存的模型
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"加载模型来自 {model_path}")
    else:
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "未提供或未找到模型路径，跳过加载。")

    # 评估原模型
    UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "原模型评估：")
    ori_loss, ori_accuracy = model.evaluate(test_loader, device=str(device))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用传入的学习率

    # 开始训练
    model.train()  # 设置模型为训练模式

    first_epoch_loss = None
    last_epoch_loss = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        # 记录每个epoch的平均损失
        avg_loss = running_loss / len(train_loader)
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if epoch == 0:
            first_epoch_loss = avg_loss
        if epoch == num_epochs - 1:
            last_epoch_loss = avg_loss

    # 如果效果比原模型好，就更新原模型
    UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "新模型评估：")
    newloss, new_accuracy = model.evaluate(test_loader, device=str(device))
    if new_accuracy > ori_accuracy:
        model.save_model(model_path)
        return model,new_accuracy
    else:
        return model,ori_accuracy


# 微调整个网络，不更新模型，用于计算数据质量
def fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=20, device='cpu', lr=1e-5,
                                    model_path=None):
    """
    微调整个网络，但是不替换模型，用于计算数据质量
    :param model: 已训练的CNN模型 (CIFAR100CNN)
    :param train_loader: 训练数据加载器，其中包含数据
    :param test_loader: 测试数据加载器
    :param num_epochs: 训练的轮数
    :param device: 计算设备 ('cpu' 或 'cuda')
    :param lr: 微调时的学习率
    :param model_path: 加载模型的路径
    :return: 微调后的单位数据 loss 差
    """

    # 将模型移动到指定设备并转换为Float类型
    model.to(device).float()

    # 加载预训练模型权重（如果有保存的模型）
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)  # 加载先前保存的模型
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"加载预训练模型来自 {model_path}")
    else:
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "未提供或未找到预训练模型路径，跳过加载。")

    # 评估原模型
    UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "原模型评估：")
    original_loss, original_accuracy = model.evaluate(test_loader, device=device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用传入的学习率

    # 开始训练
    model.train()  # 设置模型为训练模式

    first_epoch_loss = None
    last_epoch_loss = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        # 记录每个epoch的平均损失
        avg_loss = running_loss / len(train_loader)
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if epoch == 0:
            first_epoch_loss = avg_loss
        if epoch == num_epochs - 1:
            last_epoch_loss = avg_loss

    # 评估新模型
    UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "新模型评估：")
    new_loss, new_accuracy = model.evaluate(test_loader, device=device)

    # 计算 loss 差
    if first_epoch_loss is not None and last_epoch_loss is not None:
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "loss 差为：")
        loss_diff = first_epoch_loss - last_epoch_loss
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path,
                                   f"{first_epoch_loss} - {last_epoch_loss} = {loss_diff:.4f}")
        unitDataLossDiff = loss_diff / len(train_loader.dataset)
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, f"单位数据 loss 差为：{unitDataLossDiff:.6f}")
    else:
        UtilsCIFAR100.print_and_log(global_cifar100_parent_path, "未能计算 loss 差。")
        unitDataLossDiff = 0.0

    return unitDataLossDiff
