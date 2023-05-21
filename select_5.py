import sys

from pandas.core.indexes.base import F
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import pandas as pd
import numpy as np

#  load test data
internet_data = pd.read_csv('data/select_data/cache/internet.csv')

internet_data['issue_date'] = pd.to_datetime(internet_data['issue_date'], format='%Y-%m-%d')


limit = 0.7

selectd_internet_data = internet_data[internet_data['diff'] < limit].copy()

print(len(selectd_internet_data) * 1.0/len(internet_data))

selectd_internet_data = selectd_internet_data.drop(['judge', 'diff'], axis = 1)

selectd_internet_data.to_csv('data/cache/selected_internet_data.csv', index = False)
