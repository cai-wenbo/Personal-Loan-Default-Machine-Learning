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
#  print(test_data.describe())

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


concat_cols = ['employer_type', 'industry', 'censor_status', 'use', 'issue_dayofweek']

for col in concat_cols:
    train_data = concating(train_data, col)
    test_data = concating(test_data, col)

#  print(test_data.describe())
#  print(train_data.describe())


counts_1 = test_data['is_default'].value_counts()[1]
portion_1 = counts_1 / len(test_data)
print(portion_1)

#  print(test_data['sub_class'].unique())
#  print(test_data.dtypes)
#  print(train_data.dtypes)
#  print(train_data.describe())
#  print(test_data.shape[1])



#  num_features = 38
num_features = 56

#  cat_index = [16, 17, 24, 28, 30, 31, 32, 33, 34, 35, 36, 37]
cat_index = [16, 17, 24, 28, 51, 52, 53, 54, 55]
#  cat_bias = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


#  bias_cols = ['post_code', 'region', 'title']

#  for col in bias_cols:
#      train_data[col] = train_data[col] - 1
#      test_data[col] = test_data[col] - 1


num_index = list(set(range(num_features)) - set(cat_index))
num_continuous = num_features - len(cat_index)

categories_unique = []

#  for i in cat_index:
#      col = train_data.columns[i]
#      train_data[col] = train_data[col].round(0)
#      test_data[col] = test_data[col].round(0)

#  print(train_data[])

for i in cat_index:
    col = train_data.columns[i]
    #  print(train_data[col].unique())
    categories_unique.append(train_data[col].nunique())



#  split
X_train = train_data.drop('is_default', axis=1)
y_train = train_data['is_default'].astype(int)
X_test = test_data.drop('is_default', axis=1)
y_test = test_data['is_default'].astype(int)


print('ok')


from lightgbm import LGBMClassifier

import lightgbm as lgb

print(X_train.describe())

#  categorical_features = [ 'name:employer_type', 'name:industry', 'name:censor_status', 'name:use', 'name:marriage', 'name:issue_dayofweek', 'name:post_code', 'name:region', 'name:initial_list_status', 'name:early_return']
#  fill_nan_with_neg(data)

#  to_drop = ['post_code', 'region', 'earlies_credit_mon_bin',  'issue_year_bin']
#  for col in to_drop:
#      X_train = X_train.drop([col], axis = 1)
#      X_test = X_test.drop([col], axis = 1)

categorical_features = ['initial_list_status', 'app_type', 'early_return', 'work_over_10ys', 'employer_type', 'industry', 'censor_status', 'use', 'issue_dayofweek']

for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")



#  d_train = lgb.Dataset(X_train, categorical_feature=categorical_features)
#  d_test = lgb.Dataset(X_test, categorical_feature=categorical_features)

#  clf = LGBMClassifier(n_estimators= 40000,
#                       num_leaves= 3**5,
#                       learning_rate= 0.2,
#                       subsample=0.3,
#                       max_depth=6,
#                       reg_alpha=.2,
#                       reg_lambda=.2,
#                       min_split_gain=7,
#                       min_child_weight=0.2,
#                       feature_fraction= 0.3,
#                       categorical_feature = categorical_features)
#                       #  feature_name = cols,


#  clf.booster_ = clf.booster_.load_model('model/lgbm/lgbm.txt')
clf = LGBMClassifier(n_estimators= 5000,
                     #  max_bin = 50,
                     #  colsample_bytree= 0.5,
                     #  min_clild_samples=10,
                     #  class_weight='balanced',
                     #  feature_fraction_rate= 7,
                     #  boosting_type = 'dart',
                     #  scale_pos_weight = 5,
                     #  objective = 'binary',
                     Is_balance = True,
                     num_leaves= 4**5,
                     learning_rate= 0.01,
                     subsample=0.8,
                     subsample_freq= 7,
                     feature_fraction = 0.4,
                     max_depth=4,
                     reg_alpha=.2,
                     reg_lambda=.2,
                     min_child_weight=1,
                     min_split_gain=3,
                     categorical_feature = categorical_features,
                     random_state=3301)


#  clf.fit(X=X_train, y = y_train, eval_set=[(X_test, y_test)], eval_metric='auc', init_model= 'model/lgbm/lgbm.txt', early_stopping_rounds=10)
#  clf.fit(X=X_train,y = y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=1000)
#  clf.fit(X=X_train,y = y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=200)
clf.fit(X=X_train,y = y_train, eval_set=[(X_test, y_test)], eval_metric='auc')

#  clf.booster_.save_model('model/lgbm/lgbm.txt')
clf.booster_.save_model('model/lgbm/only_pub_impute_smote.txt')



