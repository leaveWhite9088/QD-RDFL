import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle

from src.qd_rdfl.datasets.IMDBDataset import IMDBDataset

# 创建TextCNN模型
class IMDBTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, num_classes=2, dropout=0.5):
        """
        初始化TextCNN模型
        :param vocab_size: 词汇表大小
        :param embed_dim: 词嵌入维度
        :param filter_sizes: 卷积核尺寸列表，如[3, 4, 5]
        :param num_filters: 每种尺寸卷积核的数量
        :param num_classes: 输出类别数量（二分类为2）
        :param dropout: dropout概率
        """
        super(IMDBTextCNN, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        
        # dropout和全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        """
        前向传播
        :param x: 输入序列，shape为[batch_size, seq_len]
        :return: 分类结果
        """
        # 词嵌入 [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(x)
        
        # 增加通道维度 [batch_size, seq_len, embed_dim] -> [batch_size, 1, seq_len, embed_dim]
        x = x.unsqueeze(1)
        
        # 对不同大小的卷积核进行卷积并池化
        conv_results = []
        for conv in self.convs:
            # 卷积 [batch_size, num_filters, seq_len-filter_size+1, 1]
            conved = F.relu(conv(x)).squeeze(3)
            
            # 最大池化 [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conved, conved.size(2)).squeeze(2)
            
            conv_results.append(pooled)
        
        # 拼接不同卷积核的结果 [batch_size, num_filters * len(filter_sizes)]
        x = torch.cat(conv_results, dim=1)
        
        # dropout和全连接
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def train_model(self, train_loader, criterion, optimizer, num_epochs=5, device='cpu', model_save_path=None):
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
        
        print(f"Training on {device}...")
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            print(f"Epoch {epoch + 1}/{num_epochs} started...")
            
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
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            # 每个 epoch 结束时输出平均损失
            print(f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(train_loader):.4f}")
        
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
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    
    def save_model(self, file_path):
        """
        保存模型
        :param file_path: 保存模型的文件路径
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        加载模型
        :param file_path: 模型文件路径
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()
        print(f"Model loaded from {file_path}")

# 使用IMDB数据集训练TextCNN
def train_imdb_textcnn(data_dir="../data/dataset/IMDB", model_save_path="../data/model/imdb_textcnn"):
    """
    使用IMDB数据集，训练TextCNN
    :param data_dir: IMDB数据集所在目录
    :param model_save_path: 模型保存路径
    :return: 训练好的模型
    """
    # 设置参数
    vocab_size = 10000
    max_length = 200
    batch_size = 64
    embed_dim = 100
    filter_sizes = [3, 4, 5]
    num_filters = 100
    num_classes = 2
    dropout = 0.5
    num_epochs = 5
    
    # 加载训练数据和创建词汇表
    print("Loading training data...")
    train_dataset = IMDBDataset(data_dir, mode='train', max_length=max_length, vocab_size=vocab_size)
    
    # 保存词汇表供测试时使用
    vocab_path = os.path.join(os.path.dirname(model_save_path), 'imdb_vocab.pkl')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(train_dataset.vocab, f)
    
    # 加载测试数据
    print("Loading test data...")
    test_dataset = IMDBDataset(data_dir, mode='test', max_length=max_length, vocab=train_dataset.vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMDBTextCNN(
        vocab_size=len(train_dataset.vocab),
        embed_dim=embed_dim,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model.train_model(train_loader, criterion, optimizer, num_epochs=num_epochs, device=str(device), model_save_path=model_save_path)
    
    # 评估模型
    model.evaluate(test_loader, device=str(device))
    
    return model

# 微调模型
def fine_tune_model(model, train_loader, test_loader, num_epochs=3, device='cpu', lr=1e-5, model_path=None):
    """
    微调TextCNN模型
    :param model: 预训练的TextCNN模型
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param num_epochs: 训练轮数
    :param device: 计算设备
    :param lr: 学习率
    :param model_path: 模型保存路径
    :return: 微调后的模型和准确率
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 加载预训练模型权重（如果有保存的模型）
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)
    
    # 评估原模型
    print("原模型评估：")
    ori_accuracy = model.evaluate(test_loader, device=str(device))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 开始训练
    model.train()
    
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
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    
    # 评估新模型
    print("新模型评估：")
    new_accuracy = model.evaluate(test_loader, device=str(device))
    
    # 如果效果比原模型好，就更新原模型
    if new_accuracy > ori_accuracy and model_path:
        model.save_model(model_path)
        return model, new_accuracy
    else:
        return model, ori_accuracy

# 计算数据质量（不更新模型）
def fine_tune_model_without_replace(model, train_loader, test_loader, num_epochs=3, device='cpu', lr=1e-5, model_path=None):
    """
    微调模型但不更新，用于计算数据质量
    :param model: 预训练的TextCNN模型
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param num_epochs: 训练轮数
    :param device: 计算设备
    :param lr: 学习率
    :param model_path: 模型路径
    :return: 单位数据loss差
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 加载预训练模型权重
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)
    
    # 评估原模型
    print("原模型评估：")
    model.evaluate(test_loader, device=str(device))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 开始训练
    model.train()
    
    first_epoch_loss = 0.0
    last_epoch_loss = 0.0
    
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
        
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # 记录第一个和最后一个epoch的损失
        if epoch == 0:
            first_epoch_loss = avg_epoch_loss
        if epoch == num_epochs - 1:
            last_epoch_loss = avg_epoch_loss
    
    # 评估新模型
    print("新模型评估：")
    model.evaluate(test_loader, device=str(device))
    
    # 计算损失差异
    loss_diff = first_epoch_loss - last_epoch_loss
    print(f"Loss差为：{loss_diff:.4f}")
    
    # 计算单位数据loss差
    unit_data_loss_diff = loss_diff / len(train_loader.dataset)
    print(f"单位数据loss差为：{unit_data_loss_diff:.6f}")
    
    return unit_data_loss_diff 