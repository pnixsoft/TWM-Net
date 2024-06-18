# InSCoN from  Qfnu Normal University
# Contributor: Baokun Song, M. Zhang
# Email: 1558860588@qfnu.edu.cn
# last modified: 20240617 21:50
#dataset download: 链接: https://pan.baidu.com/s/1oaGvMEuit74KMdms3GRI-Q 提取码: yx97 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for file_name in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file_name)
            if file_name.endswith('_0.npy'):
                data = np.load(file_path)
                self.data.append(data)
                self.labels.append(np.zeros(data.shape[0], dtype=np.int64))
            elif file_name.endswith('_1.npy'):
                data = np.load(file_path)
                self.data.append(data)
                self.labels.append(np.ones(data.shape[0], dtype=np.int64))
            elif file_name.endswith('_2.npy'):
                data = np.load(file_path)
                self.data.append(data)
                self.labels.append(np.full(data.shape[0], 2, dtype=np.int64))
            elif file_name.endswith('_3.npy'):
                data = np.load(file_path)
                self.data.append(data)
                self.labels.append(np.full(data.shape[0], 3, dtype=np.int64))

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        assert len(self.data) == len(self.labels), "数据和标签的长度不匹配"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

# 数据集路径
root_dir = '../Toolweardata'

# 创建自定义数据集
dataset = CustomDataset(root_dir)

# 设置训练集和验证集的比例
train_ratio = 0.8

val_size = int(0.2 * len(dataset))
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - test_size - val_size

# 使用 random_split 分割数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#print(train_dataset.size(), val_dataset.size(), test_dataset.size())

# 使用 DataLoader 封装训练集和验证集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.fc = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, width).permute(0, 1, 3, 2)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, width).permute(0, 1, 3, 2)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, width).permute(0, 1, 3, 2)

        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, value)

        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, width)
        out = self.fc(out)

        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class EnhancedCNNWithAttention(nn.Module):
    def __init__(self, num_classes=4):
        super(EnhancedCNNWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(ResidualBlock, 64, 128, stride=2)
        self.layer2 = self._make_layer(ResidualBlock, 128, 256, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 512, stride=2)

        self.attention = MultiHeadSelfAttention(512)

        self.feature_size = 1024 // 2 // 2 // 2 // 2
        self.fc_input_dim = 512 * self.feature_size

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def calculate_confusion_matrix_elements(preds, labels, num_classes):
    conf_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    TP = conf_matrix.diag()
    FP = conf_matrix.sum(0) - TP
    FN = conf_matrix.sum(1) - TP
    TN = conf_matrix.sum() - (FP + FN + TP)

    return TP, FP, FN, TN

# 创建模型
model = EnhancedCNNWithAttention(num_classes=4)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    TP, FP, FN, TN = calculate_confusion_matrix_elements(all_preds, all_labels, 4)
    Precision = (TP / (TP + FP)).mean().item()
    Recall = (TP / (TP + FN)).mean().item()
    F1 = 2 * Precision * Recall / (Precision + Recall)
    return epoch_loss, epoch_acc, Precision, Recall, F1

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    TP, FP, FN, TN = calculate_confusion_matrix_elements(all_preds, all_labels, 4)
    Precision = (TP / (TP + FP)).mean().item()
    Recall = (TP / (TP + FN)).mean().item()
    F1 = 2 * Precision * Recall / (Precision + Recall)
    return epoch_loss, epoch_acc, Precision, Recall, F1

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    TP, FP, FN, TN = calculate_confusion_matrix_elements(all_preds, all_labels, 4)
    Precision = (TP / (TP + FP)).mean().item()
    Recall = (TP / (TP + FN)).mean().item()
    F1 = 2 * Precision * Recall / (Precision + Recall)
    return epoch_loss, epoch_acc, Precision, Recall, F1

# 训练和验证模型
num_epochs = 50
best_val_acc = 0.0  # 用于保存最佳模型
for epoch in range(num_epochs):
    train_loss, train_acc, train_prec, train_recall, train_F1 = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_prec, val_recall, val_F1 = validate(model, val_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Prec: {train_prec*100:.2f}%, Train Recall: {train_recall*100:.2f}%, Train F1: {train_F1*100:.2f}%')
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Prec: {val_prec*100:.2f}%, Val Recall: {val_recall*100:.2f}%, Val F1: {val_F1*100:.2f}%')

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
print("Best model loaded with validation accuracy:", best_val_acc)

# 在测试集上评估最佳模型
test_loss, test_acc, test_prec, test_recall, test_F1 = test(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Prec: {test_prec*100:.2f}%, Test Recall: {test_recall*100:.2f}%, Test F1: {test_F1*100:.2f}%')