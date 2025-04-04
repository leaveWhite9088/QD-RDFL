import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.qd_rdfl.datasets.MNISTDataset import MNISTDataset
from src.qd_rdfl.utils.UtilsMNIST import UtilsMNIST
from src.qd_rdfl.global_variable import global_minst_parent_path

class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        """
        初始化CNN模型
        :param num_classes: 输出类别数量
        """
        super(MNISTCNN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出通道数调整为128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 保证池化后尺寸减半

        # 定义全连接层
        self.fc1 = nn.Linear(128 * 14 * 14, 128)  # 输入特征是 14x14x128
        self.fc2 = nn.Linear(128, num_classes)  # 输出类别数量

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 模型输出
        """
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积 + ReLU + 池化

        x = x.view(-1, 128 * 14 * 14)  # 展平，这里使用正确的展平维度
        x = F.relu(self.fc1(x))  # 全连接层1 + ReLU
        x = self.fc2(x)  # 全连接层2（输出层）
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs=5, device='cpu',
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
            UtilsMNIST.print_and_log(global_minst_parent_path,f"Epoch {epoch + 1}/{num_epochs} started...")  # 打印每个epoch的开始

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
                    UtilsMNIST.print_and_log(global_minst_parent_path,
                        f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 每个 epoch 结束时输出平均损失
            UtilsMNIST.print_and_log(global_minst_parent_path,f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(train_loader):.4f}")

        if model_save_path is not None:
            # 保存最终模型
            self.save_model(model_save_path)

    def evaluate(self, test_loader, device='cpu'):
        """
        评估模型
        :param test_loader: 数据加载器（测试集）
        :param device: 计算设备（'cpu' 或 'cuda'）
        :return: 模型在测试集上的准确率
        """
        self.to(device)
        self.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        UtilsMNIST.print_and_log(global_minst_parent_path,f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, file_path):
        """
        保存模型
        :param file_path: 保存模型的文件路径
        """
        torch.save(self.state_dict(), file_path)
        UtilsMNIST.print_and_log(global_minst_parent_path,f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        加载模型
        :param file_path: 模型文件路径
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()
        UtilsMNIST.print_and_log(global_minst_parent_path,f"Model loaded from {file_path}")


# 使用minist数据集，训练cnn
def train_minist_cnn(model_save_path="../data/model/mnist_cnn"):
    """
    使用minist数据集，训练cnn
    :param model_save_path: 保存的路径
    :return:
    """
    # 数据转换：将图像转换为Tensor并进行归一化
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_images_path = "../data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = "../data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = "../data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = "../data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_dataset = MNISTDataset(train_images_path, train_labels_path)
    test_dataset = MNISTDataset(test_images_path, test_labels_path)

    # 使用 DataLoader 加载数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                      model_save_path=model_save_path)

    # 评估模型
    model.evaluate(test_loader, device=str(device))


# 微调整个网络
def fine_tune_model(model, train_loader, test_loader, num_epochs=5, device='cpu', lr=1e-5, model_path=None):
    """
    微调整个网络
    :param model: 已训练的CNN模型
    :param train_loader: 训练数据加载器，其中包含数据
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    :param device: 计算设备 ('cpu' 或 'cuda')
    :param lr: 微调时的学习率
    :param model_save_path: 保存模型的路径，默认不保存
    :return: 微调后的模型
    """

    # 将模型移动到指定设备
    model.to(device)

    # 加载预训练模型权重（如果有保存的模型）
    model.load_model(model_path)  # 加载先前保存的模型

    # 评估模型
    UtilsMNIST.print_and_log(global_minst_parent_path,"原模型评估：")
    ori_accuracy = model.evaluate(test_loader, device=str(device))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置新的学习率（如果需要微调时设置更小的学习率）
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 开始训练
    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        UtilsMNIST.print_and_log(global_minst_parent_path,f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # 如果效果比原模型好，就更新原模型
    UtilsMNIST.print_and_log(global_minst_parent_path,"新模型评估：")
    new_accuracy = model.evaluate(test_loader, device=str(device))
    if new_accuracy > ori_accuracy:
        model.save_model(model_path)
        return model, new_accuracy
    else:
        return model, ori_accuracy



# 微调整个网络
def fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=5, device='cpu', lr=1e-5, model_path=None):
    """
    微调整个网络，但是不更新模型，用于计算数据质量
    :param model: 已训练的CNN模型
    :param train_loader: 训练数据加载器，其中包含数据
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    :param device: 计算设备 ('cpu' 或 'cuda')
    :param lr: 微调时的学习率
    :param model_save_path: 保存模型的路径，默认不保存
    :return: 微调后的模型
    """

    # 将模型移动到指定设备
    model.to(device)

    # 加载预训练模型权重（如果有保存的模型）
    model.load_model(model_path)  # 加载先前保存的模型

    # 评估模型
    UtilsMNIST.print_and_log(global_minst_parent_path,"原模型评估：")
    model.evaluate(test_loader, device=str(device))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置新的学习率（如果需要微调时设置更小的学习率）
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 开始训练
    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        UtilsMNIST.print_and_log(global_minst_parent_path,f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # 记录loss，用于计算loss差
        if epoch == 0:
            first_epoch_loss = running_loss / len(train_loader)

        if epoch == num_epochs - 1:
            last_epoch_loss = running_loss / len(train_loader)

    # # 如果指定了模型保存路径，保存微调后的模型
    # if model_save_path:
    #     model.save_model(model_save_path)

    # 如果效果比原模型好，就更新原模型
    UtilsMNIST.print_and_log(global_minst_parent_path,"新模型评估：")
    model.evaluate(test_loader, device=str(device))
    UtilsMNIST.print_and_log(global_minst_parent_path,"loss差为：")
    UtilsMNIST.print_and_log(global_minst_parent_path,first_epoch_loss - last_epoch_loss)
    UtilsMNIST.print_and_log(global_minst_parent_path,"单位数据loss差为：")
    unitDataLossDiff = (first_epoch_loss - last_epoch_loss) / len(train_loader.dataset)
    UtilsMNIST.print_and_log(global_minst_parent_path,unitDataLossDiff)

    return unitDataLossDiff