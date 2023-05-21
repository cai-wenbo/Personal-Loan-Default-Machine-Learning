
import pandas as pd
import numpy as np

from src.my_gain import train_gain
from src.my_gain import impute_data


train_data = pd.read_csv('data/cache/train_data_2.csv')

gain_parameters = {
    'batch_size': 8,
    'hint_rate': 0.8,
    'alpha': 100,
    'iterations':50000}

train_gain(train_data, gain_parameters, 'model/gain/')
