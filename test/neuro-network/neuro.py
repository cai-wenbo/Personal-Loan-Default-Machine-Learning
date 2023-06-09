import pandas as pd
import numpy as np
from pandas._libs.lib import is_float
from pandas.core.arrays import categorical

pd.options.display.max_columns = None  # 打印所有列
pd.options.display.max_rows = None     # 打印所有行

test = pd.read_csv('./input/test_public.csv')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
train1 = pd.read_csv('./input/train_public.csv')
train2 = pd.read_csv('./input/train_internet.csv')

train1['issue_date'] = pd.to_datetime(train1['issue_date'], format='%Y/%m/%d')
train2['issue_date'] = pd.to_datetime(train2['issue_date'], format='%Y-%m-%d')

data = pd.concat([train1, train2], ignore_index=True)


#  get a smaller dataset to test
proportion = 0.1
sample_size = int(len(data) * proportion)
data = data.sample(frac = proportion)

#


#preprocess

#  import re
#
#  def remove_non_numeric(string):
#       return re.sub('[^0-9\.]', '', string)

#  data['work_year'] = data['work_year'].apply(remove_non_numeric).astype(int)


#  delete irrelevant columes
data = data.drop(['loan_id','user_id','earlies_credit_mon'], axis = 1)

#  data['issue_date'] = pd.to_datetime(data['issue_date'], format='%Y/%m/%d')
data['issue_date'] = data['issue_date'].astype(int) // 86400000000000


#  print(data['earlies_credit_mon'].unique())

import re

import re
def extract_number(x):
    if not isinstance(x, str):
        return x
    numbers = re.findall(r'\d+', x)
    if len(numbers) > 0:
        return int(''.join(numbers))
    else:
        return None

# extract number of some strings 
data['work_year'] = data['work_year'].apply(extract_number)
data['class'] = data['class'].apply(lambda x: ord(x))
data['sub_class'] = data['sub_class'].apply(extract_number)


#  append blank
data = data.apply(lambda x: x.fillna(x.median()) if pd.api.types.is_numeric_dtype(x) else x)




data['work_year'] = data['work_year'].astype(int)


#number embedding

def chinese_to_int(chinese_str):
    if  isinstance(chinese_str, float):
        return 0;
    else:
        return int(str(ord(chinese_str[0]))[0])

def chinese_to_string(chinese_str):
    if  isinstance(chinese_str, float):
        return "dfas";
    else:
        s = "" 
        char1 = chr(ord('a') + int(str(ord(chinese_str[0]))[0]))
        char2 = chr(ord('a') + int(str(ord(chinese_str[0]))[1]))
        char3 = chr(ord('a') + int(str(ord(chinese_str[1]))[0]))
        char4 = chr(ord('a') + int(str(ord(chinese_str[1]))[1]))
        s = s + char1 + char2 + char3 + char4
        return s 

#  data['employer_type'] = data['employer_type'].apply(chinese_to_int)
#  data['industry'] = data['industry'].apply(chinese_to_int)
#  data['work_type'] = data['work_type'].apply(chinese_to_int)

data['employer_type'] = data['employer_type'].apply(chinese_to_string)
data['industry'] = data['industry'].apply(chinese_to_string)
data['work_type'] = data['work_type'].apply(chinese_to_string)

#one hot encodding
one_hot_employer_type = pd.get_dummies(data['employer_type'],prefix='et', dtype = int)
print(one_hot_employer_type.info())
#  one_hot_employer_type = one_hot_employer_type.add_prefix('et_')
one_hot_industry = pd.get_dummies(data['industry'], prefix='i', dtype = int)
#  one_hot_employer_type = one_hot_employer_type.add_prefix('i_')
one_hot_work_type = pd.get_dummies(data['work_type'], prefix = 'wt', dtype = int)
#  one_hot_employer_type = one_hot_employer_type.add_prefix('wt_')
data = data.drop(['employer_type', 'industry', 'work_type'], axis = 1)
data = pd.concat([data, one_hot_employer_type, one_hot_industry, one_hot_work_type], axis=1)








#  data['employer_type'] = data['employer_type'].replace({'政府机构': 1, '幼教与中小学校': 2, '普通企业': 3, '世界五百强': 4, '高等教育机构': 5})


#  preprocess block end


#  put the label to the last column
cols = list(data.columns)
cols.remove('is_default')
cols.append('is_default')
data = data.reindex(columns=cols)



counts_1 = data['is_default'].value_counts()[1]
portion_1 = counts_1 / len(data)
print(portion_1)




print('\n')
#  print(data.head)
#  print(data.shape[1])
print(data.dtypes)



# 定义特征和标签列
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=354)


#  use SMOTE

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state= 151)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())



# 将数据转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(63, 96)
        self.fc2 = nn.Linear(96, 32)
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

import math 
from sklearn.metrics import roc_auc_score


#  def BCELoss_weighted(weights)
#
#      def loss(input, target):
#          input = torch.clamp(input,min=1e-7,max=1-1e-7)
#          bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
#          return torch.mean(bce)
#
#      return loss

# 定义训练函数
def train(net, optimizer, criterion, train_loader, test_loader, epochs):
    #  n = 1;
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels)
            train_loss += loss.item()
            

            #  loss = loss / math.sqrt(n);
            #  n = n + 1;
            #  if labels.item == 1:
            #      loss = loss / portion_1
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        net.eval()
        test_loss = 0.0
        correct = 0
        tp = 0
        #  tp_fp = 0
        tp_fn = 0

        y_true_list = []
        y_score_list = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = net(inputs)
                test_loss += criterion(outputs.squeeze(), labels).item()
                predicted = torch.round(outputs.squeeze())
                correct += (predicted == labels).sum().item()

                tp += ((outputs == labels) & (labels ==1)).sum().item()
                tp_fn += (labels ==1).sum().item()
                #  tp_fp += (outputs == 1).sum().item()
                y_true_list.extend(labels.tolist())  
                y_score_list.extend(outputs.tolist())  
                
        test_loss /= len(test_loader)
        auc_roc = roc_auc_score(y_true_list, y_score_list)
                

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {correct/len(y_test):.4f}, Test Recall: {tp/tp_fn:.4f}, AUC_ROC: {auc_roc:.4f}')

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化神经网络、损失函数和优化器
net = Net()
#  criterion = nn.BCEWithLogitsLoss()
#  weights = torch.where(labels == 1, torch.Tensor([2]), torch.Tensor([1]))

#  class weighting
#  class_weight = torch.Tensor([1/portion_1])
#  criterion = nn.BCELoss(weight = class_weight)

criterion = nn.BCELoss()
#  criterion.weight = weights


#  criterion = nn.BCELoss()
#  criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 1e-4)

# 开始训练
train(net, optimizer, criterion, train_loader, test_loader, epochs=500)
