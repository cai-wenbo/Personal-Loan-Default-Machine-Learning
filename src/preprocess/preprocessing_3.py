import pandas as pd
import numpy as np



#  train_data = pd.read_csv('../../train_data_2.csv')
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
#  train_data.to_csv('../../train_data_3.csv', index = False)
