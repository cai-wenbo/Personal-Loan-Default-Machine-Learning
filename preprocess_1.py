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

sys.path.append('./src')


import threading
lock = threading.Lock()

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




#  load data
train_pub = pd.read_csv('data/raw_data/train_public.csv')
train_int = pd.read_csv('data/raw_data/train_internet.csv')

train_pub['issue_date'] = pd.to_datetime(train_pub['issue_date'], format='%Y/%m/%d')
train_int['issue_date'] = pd.to_datetime(train_int['issue_date'], format='%Y-%m-%d')

#  data = train1
#  data = pd.concat([train1, train2])

#  get a smaller dataset to test

#  sample_size = int(len(train_pub) * proportion)
#  train_pub = train_pub.sample(frac = proportion)


proportion = 0.1
sample_size = int(len(train_int) * proportion)
train_int = train_int.sample(frac = proportion)


#split to train_data and test_data

test_proportion = 0.1
test_size = int(len(train_pub) * test_proportion)

np.random.seed(3301)
test_indices = np.random.choice(train_pub.index, size=test_size, replace=False)

test_data = train_pub[train_pub.index.isin(test_indices)]
train_pub = train_pub[~train_pub.index.isin(test_indices)]

lock.acquire()

train_int.to_csv('data/cache/train_int_1.csv', index = False)
train_pub.to_csv('data/cache/train_pub_1.csv', index = False)
test_data.to_csv('data/cache/test_data_1.csv', index = False)

lock.release()



#  process train_pub and store some parameters
train_pub = convert(train_pub)
train_pub = clean(train_pub)
train_pub =  feature_extract(train_pub)

#  train_pub, bin_edges = binning(train_pub)


import pickle
#  with open('data/analysis_data/bin_edges.pkl', 'wb') as f:
#      pickle.dump(bin_edges, f)

train_pub = encoding(train_pub)

get_embedding_dicts(train_pub, 'data/analysis_data/saved_dicts.pkl')
train_pub = embedding(train_pub, 'data/analysis_data/saved_dicts.pkl')

#  cols = train_pub.columns.tolist()
#  print(cols)
#  with open('data/analysis_data/cols.txt', 'w') as f:
#      f.write(','.join(cols))



#  preprocess internet data
train_int = convert(train_int)
train_int = clean(train_int)
train_int =  feature_extract(train_int)

#  train_int, _ = binning(train_int, 'data/analysis_data/bin_edges.pkl')

train_int = encoding(train_int)
train_int = embedding(train_int, 'data/analysis_data/saved_dicts.pkl')
#  train_int = aligning(train_int, 'data/analysis_data/cols.txt')



#Normalize
train_data = pd.concat([train_pub, train_int])

cols_tmp = list(train_data.columns)
cols_tmp.remove('is_default')
cols_tmp.append('is_default')
train_data = train_data.reindex(columns=cols_tmp)



#  print(train_data.dtypes)
_, norm_parameters = normalization(train_data)


cols = train_data.columns.tolist()
#  print(cols)
with open('data/analysis_data/cols.txt', 'w') as f:
    f.write(','.join(cols))

#  cols_pub = train_data.columns.tolist()
#  #  print(cols)
#  with open('data/analysis_data/cols_pub.txt', 'w') as f:
#      f.write(','.join(cols_pub))

#  print(norm_parameters)
with open('data/analysis_data/norm_parameters.pkl', 'wb') as f:
    pickle.dump(norm_parameters, f)


cols_pub = list(train_pub.columns)
cols_int = list(train_int.columns)

with open('data/analysis_data/cols_pub.txt', 'w') as f:
    f.write(','.join(cols_pub))
with open('data/analysis_data/cols_int.txt', 'w') as f:
    f.write(','.join(cols_int))

print(train_pub.dtypes)

train_pub = aligning(train_pub, 'data/analysis_data/cols.txt')
train_int = aligning(train_int, 'data/analysis_data/cols.txt')

print(train_pub.dtypes)

train_pub, _ = normalization(train_pub, norm_parameters)
train_int, _ = normalization(train_int, norm_parameters)

train_pub = train_pub.reindex(columns = cols_pub)
train_int = train_int.reindex(columns = cols_int)

print(train_pub.dtypes)


#
train_pub.to_csv('data/cache/train_pub_2.csv', index = False)
train_int.to_csv('data/cache/train_int_2.csv', index = False)

#
#
