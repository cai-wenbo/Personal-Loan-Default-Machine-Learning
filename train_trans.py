import sys

sys.path.append('./src')

import pandas as pd
import numpy as np
pd.options.display.max_rows = None
pd.set_option('display.max_columns', None)


from preprocessing import convert, embedding
from preprocessing import clean
from preprocessing import feature_extract
from preprocessing import binning
from preprocessing import encoding
from preprocessing import embedding
from preprocessing import get_category_dict
from preprocessing import get_embedding_dicts
from preprocessing import get_category_proportions
from preprocessing import aligning
from preprocessing import normalization
from my_gain import train_gain
from my_gain import impute_data
#  from neuro import Net, train_neuro
from filter import filte
from sklearn.preprocessing import StandardScaler 

from functools import update_wrapper
import sys
from scipy.sparse import data


import torch
import torch.nn as nn
import torch.optim as optim



#  load data
#  train_data = pd.read_csv('data/cache/train_data_4.csv')

test_data = pd.read_csv('data/cache/test_data_1.csv')

#  train_data = filte(train_data, 'model/filter/public_filter.pt', 0.05)
#  train_pub = pd.read_csv('data/cache/train_pub_4.csv')
#  train_int = pd.read_csv('data/cache/train_int_4.csv')
#  train_data = pd.concat([train_int, train_pub])
#  train_data = pd.concat([train_int, train_pub])
train_data = pd.read_csv('data/cache/train_pub_4.csv')
#  train_data = pd.read_csv('data/cache/train_int_3.csv')

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
#  test_data, bin_edges = binning(test_data, 'data/analysis_data/bin_edges.pkl')
test_data = encoding(test_data)
test_data = embedding(test_data, 'data/analysis_data/saved_dicts.pkl')
test_data = aligning(test_data, 'data/analysis_data/cols.txt')

import pickle
with open('data/analysis_data/norm_parameters.pkl', 'rb') as f:
    norm_parameters = pickle.load(f)

test_data, norm_parameters = normalization(test_data, norm_parameters)

#  print(test_data.shape[1])

test_data = impute_data(test_data, 'model/gain_pub_and_int')

test_data = aligning(test_data, 'data/analysis_data/cols_pub.txt')
train_data = aligning(train_data, 'data/analysis_data/cols_pub.txt')

def concating(df1, col):
    df = df1.copy()
    cols = df.filter(regex="^"+col).columns
    df[col] = df[cols].idxmax(axis=1).str.split("_").str.get(-1).astype(int)
    for col_ in cols:
        df = df.drop([col_], axis = 1)

    #  put the label to the last column
    cols = list(df.columns)
    cols.remove('is_default')
    cols.append('is_default')
    df = df.reindex(columns=cols)

    return df


concat_cols = ['employer_type', 'industry', 'censor_status', 'use', 'issue_dayofweek', 'post_code', 'region', 'title']
#  concat_cols = ['employer_type', 'industry', 'censor_status', 'use', 'issue_dayofweek']

for col in concat_cols:
    train_data = concating(train_data, col)
    test_data = concating(test_data, col)

#  print(test_data.describe())
#  print(train_data.describe())


counts_1 = test_data['is_default'].value_counts()[1]
portion_1 = counts_1 / len(test_data)
#  print(portion_1)

#  print(test_data['sub_class'].unique())
#  print(test_data.dtypes)
#  print(train_data.dtypes)
#  print(train_data.describe())
#  print(test_data.shape[1])
#  print(train_data.dtypes)



num_features = 38
#  num_features = 56

cat_index = [16, 17, 24, 28, 30, 31, 32, 33, 34, 35, 36, 37]
#  cat_index = [16, 17, 24, 28, 51, 52, 53, 54, 55]
#  cat_bias = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

bias_cols = ['post_code', 'region', 'title']

for col in bias_cols:
    train_data[col] = train_data[col] - 1
    test_data[col] = test_data[col] - 1


num_index = list(set(range(num_features)) - set(cat_index))

#  print(num_index)
num_continuous = num_features - len(cat_index)

categories_unique = []

for i in cat_index:
    col = train_data.columns[i]
    train_data[col] = train_data[col].round(0)
    test_data[col] = test_data[col].round(0)

#  print(train_data[])

for i in cat_index:
    col = train_data.columns[i]
    print(train_data[col].unique())
    categories_unique.append(train_data[col].nunique())

#  X = []
#  y = []
#  categories_unique = []

#  train_data.to_csv('data/cache/train_data_2.csv', index = False)
#  test_data.to_csv('data/cache/test_data_2.csv', index = False)

#  train_data = pd.read_csv('data/cache/train_data_2.csv')
#  test_data = pd.read_csv('data/cache/test_data_2.csv')



len_train = train_data.shape[0]
len_test = test_data.shape[0]


#  split
X_train = train_data.drop('is_default', axis=1)
y_train = train_data['is_default'].astype(int)
X_test = test_data.drop('is_default', axis=1)
y_test = test_data['is_default'].astype(int)

#  print(X_train.describe())
#  print(X_test.describe())

#  print('ok')


#  print(categories_unique)

#  convert to numpy
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

print(categories_unique)


scaler = StandardScaler()

for n in num_index:
    X_train[:,[n]] = scaler.fit_transform(X_train[:,n].reshape(-1, 1))
    X_test[:,[n]] = scaler.fit_transform(X_test[:,n].reshape(-1, 1))



#  #  convert to tensor
X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
y_train, y_test = torch.tensor(y_train).float(), torch.tensor(y_test).float()
#  X_train = torch.FloatTensor(X_train)
#  X_test = torch.FloatTensor(X_test)
#  y_train = torch.FloatTensor(y_train)
#  y_test = torch.FloatTensor(y_test)


dim = 50
depth = 6
heads = 8
dropout = 0.1
batch_size = 20
learning_rate = 0.000003
weight_decay = 1e-5
epochs = 30


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tab_transformer_pytorch import FTTransformer

print(sum(categories_unique))
model = FTTransformer(

            categories = categories_unique,      # tuple containing the number of unique values within each category
            num_continuous = num_continuous,                # number of continuous values
            dim = dim,                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = depth,                          # depth, paper recommended 6
            heads = heads,                          # heads, paper recommends 8
            attn_dropout = dropout,                 # post-attention dropout
            ff_dropout = dropout                    # feed forward dropout
        )

import os

file_path = 'model/transformer/transformer.pt'
if os.path.exists(file_path):
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)



optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

criterion = nn.BCEWithLogitsLoss()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


min_val_loss = float("inf")  # a big value
min_val_loss_idx = 0

from sklearn.metrics import roc_auc_score

for epoch in range(epochs):

    train_loss = 0.0

    for i, (batch_X, batch_y) in enumerate(train_loader):

        optimizer.zero_grad()

        x_categ = batch_X[:, cat_index].int()
        x_cont = batch_X[:, num_index]

        #  print(x_categ)
        #  print(x_categ.shape)
        #  print(x_cont.shape)
        out = model(x_categ, x_cont)
        # out = model( x_cont,x_categ)

        out = out.squeeze()

        loss = criterion(out, batch_y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()


    #  print(train_loss)
    train_loss /= len_train
    #  train_loss /= len(train_loader.dataset)

    model.eval()


    test_loss = 0.0
    correct = 0
    tp = 0
    #  tp_fp = 0
    tp_fn = 0
    y_true_list = []
    y_score_list = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            #  if has_nan(inputs) or has_nan(labels):
            x_categ = batch_X[:, cat_index].int()
            x_cont = batch_X[:, num_index]

            out = model(x_categ, x_cont)
            out_s = out.squeeze()
            #  print(labels)
            #  print(outputs)

            #  print(outputs)
            #  print(labels)

            test_loss += criterion(out_s, batch_y).item()
            out = out.sigmoid()
            out = out.squeeze()
            predict = torch.round(out)

            correct += (predict == batch_y).sum().item()
            tp += ((predict == batch_y) & (batch_y ==1)).sum().item()
            tp_fn += (batch_y ==1).sum().item()
            #  tp_fp += (outputs == 1).sum().item()

            y_true_list.extend(batch_y.tolist())
            y_score_list.extend(out.tolist())

    test_loss /= len_test
    test_acc = correct / len_test
    auc_roc = roc_auc_score(y_true_list, y_score_list)
    #  print(tp)
    #  print(tp_fn)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {tp/tp_fn:.4f}, AUC_ROC: {auc_roc:.4f}')



torch.save(model.state_dict(), 'model/transformer/transformer.pt')


#  # create dataloaders
#  from torch.utils.data import DataLoader
#  from torch.utils.data import TensorDataset

#  train_dataset = TensorDataset(X_train, y_train)
#  train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#  test_dataset = TensorDataset(X_test, y_test)
#  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#  #  load net
#  net = Net()

#  import os

#  file_path = 'model/neuro_network/neuro_network.pt'
#  if os.path.exists(file_path):
#      state_dict = torch.load(file_path)
#      net.load_state_dict(state_dict)



#  #def criterion
#  class_weight = torch.Tensor([5])
#  criterion = nn.BCELoss(weight = class_weight)
#  #  criterion = nn.BCELoss()

#  #  def optimizer
#  optimizer = optim.Adam(net.parameters(), lr=0.000005, weight_decay = 7e-7)


#  # start train
#  train_neuro(net, optimizer, criterion, train_loader, test_loader, epochs=50, l1_weight = 45e-4)


#  #
#  torch.save(net.state_dict(), 'model/neuro_network/neuro_network.pt')





