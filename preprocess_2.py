#  #train GAIN
import pandas as pd
import numpy as np

from src.preprocess.my_gain import train_gain
from src.preprocess.preprocessing_2 import preprocess_4

#  load data

train_data= pd.read_csv('data/cache/train_data_1.csv')





#impute train data
train_data = preprocess_4(train_data)

    


#save data
train_data.to_csv('data/cache/train_data_2.csv', index = False)


#  #use SMOTE to oversample

#  train_data = pd.read_csv('train_data_2.csv')


#  split
X_train = train_data.drop("is_default", axis=1)
y_train = train_data["is_default"]  
print(y_train.unique())



#  use SMOTE
#  from imblearn.over_sampling import SMOTE
#  sm = SMOTE(random_state= 151)
#  X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train, columns=["is_default"])

#  concat
train_data = pd.concat([X_train, y_train], axis = 1)
train_data.to_csv('data/cache/train_data_3.csv', index = False)
