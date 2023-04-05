import pandas as pd
import numpy as np


test = pd.read_csv('./input/test_public.csv')
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
train1 = pd.read_csv('./input/train_public.csv')
train2 = pd.read_csv('./input/train_internet.csv')
data = pd.concat([train1, train2], ignore_index=True)


#preprocess
import re

def remove_non_numeric(string):
     return re.sub('[^0-9\.]', '', string)

data['work_year'] = data['work_year'].apply(remove_non_numeric).astype(int)


# 定义特征和标签列
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(35, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# 定义训练函数
def train(net, optimizer, criterion, train_loader, test_loader, epochs):
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        net.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = net(inputs)
                test_loss += criterion(outputs.squeeze(), labels).item()
                predicted = torch.round(outputs.squeeze())
                correct += (predicted == labels).sum().item()
            test_loss /= len(test_loader)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {correct/len(y_test):.4f}')

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化神经网络、损失函数和优化器
net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练
train(net, optimizer, criterion, train_loader, test_loader, epochs=10)

