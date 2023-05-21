import sys

from pandas.core.indexes.base import F
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import pandas as pd
import numpy as np

#  load test data
test_data = pd.read_csv('./input/test_public.csv')
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'], format='%Y/%m/%d')
#  test_data['issue_date'] = test_data['issue_date'].astype(int) // 86400000000000

#preprocess test_data

from src.preprocess.preprocessing_1 import preprocess_1
from src.preprocess.preprocessing_1 import preprocess_2
from src.preprocess.preprocessing_1 import preprocess_3
from src.preprocess.preprocessing_2 import preprocess_4

from src.preprocess.my_gain import impute_data

result_id = test_data['loan_id']

test_data = preprocess_1(test_data)
test_data = preprocess_2(test_data, 'data/analysis_data/saved_dicts.pkl')
test_data = preprocess_3(test_data, 'data/analysis_data/cols.txt')
test_data = preprocess_4(test_data, 'data/analysis_data/norm_parameters.csv', 'model/gain')

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
result = classifier(test_data, 'data/analysis_data/norm_parameters.csv', 'model/neuro_network/neuro_network.pt')

#  print(result)

#save result
output = pd.DataFrame()
output['id'] = result_id
#  print(result)
output['isDefault'] = result['is_default']
output.to_csv('submit.csv', index = False)
