import sys

sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import pandas as pd
import numpy as np
pd.options.display.max_rows = None
pd.set_option('display.max_columns', None)


from src.preprocessing import convert, embedding
from src.preprocessing import clean
from src.preprocessing import feature_extract
from src.preprocessing import binning
from src.preprocessing import encoding
from src.preprocessing import embedding
from src.preprocessing import get_category_dict
from src.preprocessing import get_embedding_dicts
from src.preprocessing import get_category_proportions
from src.preprocessing import aligning
from src.preprocessing import normalization
from src.my_gain import train_gain
from src.my_gain import impute_data
from src.neuro import Net, train_neuro
from src.filter import filte

import torch
import torch.nn as nn
import torch.optim as optim


#  load data
train_data = pd.read_csv('data/cache/train_data_4.csv')
test_data = pd.read_csv('data/cache/test_data_1.csv')

#  train_data = filte(train_data, 'model/filter/public_filter.pt', 0.05)

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

test_data = convert(test_data)
test_data = clean(test_data)
test_data =  feature_extract(test_data)
test_data, bin_edges = binning(test_data, 'data/analysis_data/bin_edges.pkl')
test_data = encoding(test_data)
test_data = embedding(test_data, 'data/analysis_data/saved_dicts.pkl')
test_data = aligning(test_data, 'data/analysis_data/cols.txt')

import pickle
with open('data/analysis_data/norm_parameters.pkl', 'rb') as f:
    norm_parameters = pickle.load(f)
test_data, norm_parameters = normalization(test_data, norm_parameters)
test_data = impute_data(test_data, 'model/gain')

counts_1 = test_data['is_default'].value_counts()[1]
portion_1 = counts_1 / len(test_data)
print(portion_1)

#  print(test_data['sub_class'].unique())

#  split
X_train = train_data.drop('is_default', axis=1)
y_train = train_data['is_default'].astype(int)
X_test = test_data.drop('is_default', axis=1)
y_test = test_data['is_default'].astype(int)  

#  print(X_train.describe())
#  print(X_test.describe())

print('ok')



#  convert to numpy
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()



#  convert to tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# create dataloaders
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
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
#  class_weight = torch.Tensor([10])
#  criterion = nn.BCELoss(weight = class_weight)
criterion = nn.BCELoss()

#  def optimizer
optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay = 7e-5)
            

# start train
train_neuro(net, optimizer, criterion, train_loader, test_loader, epochs=10, l1_weight = 95e-6)


#
torch.save(net.state_dict(), 'model/neuro_network/neuro_network.pt')
