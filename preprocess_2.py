#  #train GAIN
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
from src.filter import filte

import threading
lock = threading.Lock()

#  load data

train_data= pd.read_csv('data/cache/train_data_2.csv')

#impute train data
train_data = impute_data(train_data, 'model/gain/')

#  train_data = filte(train_data, 'model/filter/public_filter.pt', 0.1)

    
print(train_data.describe())


lock.acquire()
#save data
train_data.to_csv('data/cache/train_data_3.csv', index = False)

lock.release()

#  #use SMOTE to oversample

#  train_data = pd.read_csv('train_data_2.csv')


#  split
X_train = train_data.drop("is_default", axis=1)
y_train = train_data["is_default"]  
#  print(y_train.unique())



#  use SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state= 151)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train, columns=["is_default"])

#  concat
train_data = pd.concat([X_train, y_train], axis = 1)
train_data.to_csv('data/cache/train_data_4.csv', index = False)
