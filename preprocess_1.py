#  preprocessing
#      split to train data and test data
#
#      use train data to decide how build embedding mode
#          embed train data
#      use train data to train GAIN
#          use GAIN to imputate train data
#      use SMOTE for over sampling
#
#      proprocess test data
import sys
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')
import threading
lock = threading.Lock()

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
from my_gain import train_gain
from my_gain import impute_data




#  load data
train1 = pd.read_csv('data/raw_data/train_public.csv')
#  train2 = pd.read_csv('data/raw_data/train_internet.csv')
#  train2 = pd.read_csv('data/cache/selected_internet_data.csv')

train1['issue_date'] = pd.to_datetime(train1['issue_date'], format='%Y/%m/%d')
#  train2['issue_date'] = pd.to_datetime(train2['issue_date'], format='%Y-%m-%d')

data = train1
#  data = pd.concat([train1, train2])

#  get a smaller dataset to test
proportion = 1
sample_size = int(len(data) * proportion)
data = data.sample(frac = proportion)



#split to train_data and test_data

test_proportion = 0.1
test_size = int(len(data) * test_proportion)

np.random.seed()
test_indices = np.random.choice(data.index, size=test_size, replace=False)

train_data = data[~data.index.isin(test_indices)]
test_data = data[data.index.isin(test_indices)]

lock.acquire()

train_data.to_csv('data/cache/train_data_1.csv', index = False)
test_data.to_csv('data/cache/test_data_1.csv', index = False)

lock.release()

train_data = convert(train_data)
train_data = clean(train_data)
train_data =  feature_extract(train_data)
#  train_data, bin_edges = binning(train_data)

import pickle
#  with open('data/analysis_data/bin_edges.pkl', 'wb') as f:
#      pickle.dump(bin_edges, f)
#  train_data = encoding(train_data)

#  get_embedding_dicts(train_data, 'data/analysis_data/saved_dicts.pkl')
#  train_data = embedding(train_data, 'data/analysis_data/saved_dicts.pkl')

cols = train_data.columns.tolist()
print(cols)
#  with open('data/analysis_data/cols.txt', 'w') as f:
#      f.write(','.join(cols))
#
#  train_data, norm_parameters = normalization(train_data)
#  #  print(norm_parameters)
#  with open('data/analysis_data/norm_parameters.pkl', 'wb') as f:
#      pickle.dump(norm_parameters, f)
#
#  train_data.to_csv('data/cache/train_data_2.csv', index = False)
#
#
