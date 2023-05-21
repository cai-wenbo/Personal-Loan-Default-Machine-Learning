#select the data in internet dataset that have the same pattern as public

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

from src.preprocess.preprocessing_1 import preprocess_1
from src.preprocess.preprocessing_1 import get_category_proportions
from src.preprocess.preprocessing_1 import get_category_dict
from src.preprocess.preprocessing_1 import preprocess_2
from src.preprocess.preprocessing_1 import preprocess_3




#  load data
train1 = pd.read_csv('data/raw_data/train_public.csv')
#  train2 = pd.read_csv('data/raw_data/train_internet.csv')

train1['issue_date'] = pd.to_datetime(train1['issue_date'], format='%Y/%m/%d')
#  train2['issue_date'] = pd.to_datetime(train2['issue_date'], format='%Y-%m-%d')

#  data = pd.concat([train1, train2], ignore_index=True)
data = train1
#  data['issue_date'] = data['issue_date'].astype(int) // 86400000000000

#  get a smaller dataset to test
proportion = 1
sample_size = int(len(data) * proportion)
data = data.sample(frac = proportion)



#split to train_data and test_data

test_proportion = 0.1
test_size = int(len(data) * test_proportion)

test_indices = np.random.choice(data.index, size=test_size, replace=False)

train_data = data[~data.index.isin(test_indices)]
test_data = data[data.index.isin(test_indices)]


train_data = preprocess_1(train_data)


#  get proportions and dicts for embedding
issue_year_proportions = get_category_proportions(train_data, 'issue_year')
issue_dayofweek_proportions = get_category_proportions(train_data, 'issue_dayofweek')
use_proportions = get_category_proportions(train_data, 'use')
post_code_proportions = get_category_proportions(train_data, 'post_code')
region_proportions = get_category_proportions(train_data, 'region')
title_proportions = get_category_proportions(train_data, 'title')
earlies_credit_mon_proportions = get_category_proportions(train_data, 'earlies_credit_mon')

issue_year_dict = get_category_dict(train_data, 'issue_year', 'is_default')
issue_dayofweek_dict = get_category_dict(train_data, 'issue_dayofweek', 'is_default')
use_dict = get_category_dict(train_data, 'use', 'is_default')
post_code_dict = get_category_dict(train_data, 'post_code', 'is_default')
region_dict = get_category_dict(train_data, 'region', 'is_default')
title_dict = get_category_dict(train_data, 'title', 'is_default')
earlies_credit_mon_dict = get_category_dict(train_data, 'earlies_credit_mon', 'is_default')

#save dict and proportions
import pickle
with open('data/select_data/analysis_data/saved_dicts.pkl', 'wb') as f:
    pickle.dump(issue_year_proportions, f)
    pickle.dump(issue_dayofweek_proportions, f)
    pickle.dump(use_proportions, f)
    pickle.dump(post_code_proportions, f)
    pickle.dump(region_proportions, f)
    pickle.dump(title_proportions, f)
    pickle.dump(earlies_credit_mon_proportions, f)

    pickle.dump(issue_year_dict, f)
    pickle.dump(issue_dayofweek_dict, f)
    pickle.dump(use_dict, f)
    pickle.dump(post_code_dict, f)
    pickle.dump(region_dict, f)
    pickle.dump(title_dict, f)
    pickle.dump(earlies_credit_mon_dict, f)


train_data = preprocess_2(train_data, 'data/select_data/analysis_data/saved_dicts.pkl')

#get and save the order of fields
cols = train_data.columns.tolist()
with open('data/select_data/analysis_data/cols.txt', 'w') as f:
    f.write(','.join(cols))


#  save data
train_data.to_csv('data/select_data/cache/train_data_1.csv', index = False)
test_data.to_csv('data/select_data/cache/test_data_1.csv', index = False)


#  #train GAIN
from src.preprocess.my_gain import train_gain
from src.preprocess.preprocessing_2 import preprocess_4

#  load data

train_data= pd.read_csv('data/select_data/cache/train_data_1.csv')
data_x = train_data.drop(['loan_id', 'user_id'], axis= 1)

data_x = data_x.to_numpy()

gain_parameters = {
    'batch_size': 32,
    'hint_rate': 0.8,
    'alpha': 100,
    'iterations': 100000}

#  train gain and save the generator
lock.acquire()

train_gain(data_x, gain_parameters, 'model/select/gain/gain.ckpt', 'data/select_data/analysis_data/norm_parameters.csv')
lock.release()

#  #impute train data
#  lock.acquire()
#  train_data = preprocess_4(train_data)
#  lock.release()
#
#
#  #save data
#  train_data.to_csv('train_data_2.csv', index = False)
#
#
#  #  #use SMOTE to oversample
#
#  train_data = pd.read_csv('train_data_2.csv')
#
#
#  #  split
#  X_train = train_data.drop("is_default", axis=1)
#  y_train = train_data["is_default"]
#
#
#
#  #  use SMOTE
#  from imblearn.over_sampling import SMOTE
#  sm = SMOTE(random_state= 151)
#  X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
#
#  X_train = pd.DataFrame(X_train)
#  y_train = pd.DataFrame(y_train, columns=["is_default"])
#
#  #  concat
#  train_data = pd.concat([X_train, y_train], axis = 1)
#  train_data.to_csv('train_data_3.csv', index = False)
#
#
