import sys

sys.path.append('./src')

#  #train GAIN
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

import threading
lock = threading.Lock()

#  load data

train_pub = pd.read_csv('data/cache/train_pub_2.csv')
train_int = pd.read_csv('data/cache/train_int_2.csv')
train_data = pd.concat([train_pub, train_int])

#  train_pub = aligning(train_pub, 'data/analysis_data/cols.txt')
#  train_int = aligning(train_int, 'data/analysis_data/cols.txt')


#impute train data

train_data = impute_data(train_data, 'model/gain_pub_and_int/')

#  train_data_i['is_default'] = label


#  train_data = filte(train_data, 'model/filter/public_filter.pt', 0.1)


#split
train_pub = train_data.iloc[:len(train_pub), :]
train_int = train_data.iloc[len(train_pub):, :]
#  train_pub = impute_data(train_pub, 'model/gain_pub_and_int/')
#  train_int = impute_data(train_int, 'model/gain_pub_and_int/')

#  print(train_pub.describe())

train_pub = aligning(train_pub, 'data/analysis_data/cols_pub.txt')
train_int = aligning(train_int, 'data/analysis_data/cols_pub.txt')
#  print(train_pub.describe())

#  #  print(train_pub.describe())
#  #  print(train_int.describe())


lock.acquire()
#save data
#  train_data.to_csv('data/cache/train_data_3.csv', index = False)
train_pub.to_csv('data/cache/train_pub_3.csv', index = False)
train_int.to_csv('data/cache/train_int_3.csv', index = False)

lock.release()

