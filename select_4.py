import sys

from pandas.core.indexes.base import F
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import pandas as pd
import numpy as np

#  load test data
internet_data = pd.read_csv('data/raw_data/train_internet.csv')
internet_data['issue_date'] = pd.to_datetime(internet_data['issue_date'], format='%Y-%m-%d')
#  test_data['issue_date'] = test_data['issue_date'].astype(int) // 86400000000000
test_data = internet_data

#preprocess test_data

from src.preprocess.preprocessing_1 import preprocess_1
from src.preprocess.preprocessing_1 import preprocess_2
from src.preprocess.preprocessing_1 import preprocess_3
from src.preprocess.preprocessing_2 import preprocess_4

from src.preprocess.my_gain import impute_data


test_data = preprocess_1(test_data)
test_data = preprocess_2(test_data, 'data/analysis_data/saved_dicts.pkl')
test_data = preprocess_3(test_data, 'data/analysis_data/cols.txt')
test_data = preprocess_4(test_data, 'data/analysis_data/norm_parameters.csv','model/gain_allin')
test_data = preprocess_3(test_data, 'data/select_data/analysis_data/cols.txt')

#  #convert to tensor
#  import torch
#  X = torch.FloatTensor(test_data)
#
#  # create dataloaders
#  from torch.utils.data import DataLoader
#  from torch.utils.data import TensorDataset
#  X = TensorDataset(X)
#  data_loader = DataLoader(X, batch_size=32, shuffle=False)
#
#
#  #  load net
#  from src.neuro_network.neuro import Net
#  net = Net()
#  state_dict = torch.load('model/neuro_network.pt')
#  net.load_state_dict(state_dict)
#
#calculate
from src.neuro_network.neuro import classifier
result = classifier(test_data, 'data/select_data/analysis_data/norm_parameters.csv', 'model/select/neuro_network/neuro_network.pt')

#  print(result)

result = result.rename(columns={'is_default': 'judge'})
internet_data = pd.concat([internet_data, result], axis = 1)
internet_data['diff'] = abs(internet_data['judge'] - internet_data['is_default'])

#  limit = 0.5

#  selectd_internet_data = internet_data[internet_data['diff'] < limit].copy()
#
#  print(len(selectd_internet_data) * 1.0/len(internet_data))
#
#  selectd_internet_data = selectd_internet_data.drop(['judge', 'diff'], axis = 1)
#
internet_data.to_csv('data/select_data/cache/internet.csv', index = False)
