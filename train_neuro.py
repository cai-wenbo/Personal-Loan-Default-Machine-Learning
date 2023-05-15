import sys

sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import pandas as pd
import numpy as np
pd.options.display.max_rows = None


from src.preprocess.my_gain import train_gain
from src.preprocess.preprocessing_1 import preprocess_1
from src.preprocess.preprocessing_1 import preprocess_2
from src.preprocess.preprocessing_1 import preprocess_3
from src.preprocess.preprocessing_2 import preprocess_4
from src.preprocess.utils import normalization
from src.neuro_network.neuro import train
from src.neuro_network.neuro import Net

import torch
import torch.nn as nn
import torch.optim as optim


#  load data
train_data = pd.read_csv('data/cache/train_data_3.csv')
test_data = pd.read_csv('data/cache/test_data_1.csv')
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'], format='%Y-%m-%d')


#preprocess test_data
#  from src.preprocess.my_gain import impute_data
#  def preprocess_4(df):
#      data_x = df.to_numpy()
#      imputed_data = impute_data(data_x)
#      imputed_data = pd.DataFrame(imputed_data)
#
#      imputed_data.columns = df.columns
#      df.update(imputed_data)
#      return df

test_data = preprocess_1(test_data)
test_data = preprocess_2(test_data)
test_data = preprocess_3(test_data)
test_data = preprocess_4(test_data)

counts_1 = test_data['is_default'].value_counts()[1]
portion_1 = counts_1 / len(test_data)
print(portion_1)

#  print(test_data['sub_class'].unique())

#  split
X_train = train_data.drop('is_default', axis=1)
y_train = train_data['is_default'].astype(int)
X_test = test_data.drop('is_default', axis=1)
y_test = test_data['is_default'].astype(int)  

#  convert to numpy
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#  #  normalize X

#  load parameters
parameters = pd.read_csv('data/analysis_data/norm_parameters.csv')
parameters = parameters[:-1]
parametras = parameters.to_numpy()

X_train , norm_parameters = normalization(X_train, parameters)
X_test , norm_parameters = normalization(X_test, parameters)

print('ok')
#  print(np.unique(X_train['total_amount']))


#
#  print(X_train.dtypes)
#  print(y_train.unique())
#  print(X_test.dtypes)
#  print(y_test.unique())
#
#  print(X_test.head())





#  convert to tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# create dataloaders
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#  load net
net = Net()

import os

file_path = 'model/neuro_network/neuro_network.pt'
if os.path.exists(file_path):
    state_dict = torch.load(file_path)
    net.load_state_dict(state_dict)



#def criterion
criterion = nn.BCELoss()

#  def optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay = 1e-3)
            

# start train
train(net, optimizer, criterion, train_loader, test_loader, epochs=10, l1_weight = 22e-5)


#
torch.save(net.state_dict(), 'model/neuro_network/neuro_network.pt')
