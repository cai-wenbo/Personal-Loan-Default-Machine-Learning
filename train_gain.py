import sys

sys.path.append('./src')

import pandas as pd
import numpy as np

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
from preprocessing import aligning


#  train_data = pd.read_csv('data/cache/train_pub_2.csv')

train_pub = pd.read_csv('data/cache/train_pub_2.csv')
train_int = pd.read_csv('data/cache/train_int_2.csv')




train_test = pd.read_csv('data/raw_data/test_public.csv')
train_test['issue_date'] = pd.to_datetime(train_test['issue_date'], format='%Y/%m/%d')



train_test = convert(train_test)
train_test = clean(train_test)
train_test =  feature_extract(train_test)
#  train_test, _ = binning(train_test, 'data/analysis_data/bin_edges.pkl')

train_test = encoding(train_test)
train_test = embedding(train_test, 'data/analysis_data/saved_dicts.pkl')
#  train_test = aligning(train_test, 'data/analysis_data/cols.txt')

train_test = aligning(train_test, 'data/analysis_data/cols.txt')

import pickle
with open('data/analysis_data/norm_parameters.pkl', 'rb') as f:
    norm_parameters = pickle.load(f)

train_test, norm_parameters = normalization(train_test, norm_parameters)


train_test = aligning(train_test, 'data/analysis_data/cols_pub.txt')

train_data = pd.concat([train_int, train_pub, train_test])

train_data = aligning(train_data, 'data/analysis_data/cols.txt')



gain_parameters = {
    'batch_size': 8,
    'hint_rate': 0.8,
    'alpha': 100,
    'iterations':10000}

train_gain(train_data, gain_parameters, 'model/gain_pub_and_int/')
