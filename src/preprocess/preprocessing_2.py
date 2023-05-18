#  load data
#     load
#
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
#
#  train nn
#      train
#
#  save nn
#      save
#
#  use model
#      preprocess data
#      put to nn
#      save output

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from scipy.sparse import data
sys.path.append('/home/null/project/py/pattern_recognition/Personal-Loan-Default-Machine-Learning/src')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import pandas as pd
import numpy as np


from tqdm import tqdm

from src.preprocess.my_gain import generator
from src.preprocess.my_gain import train_gain
from src.preprocess.my_gain import impute_data

from src.preprocess.utils import normalization, renormalization, rounding
from src.preprocess.utils import xavier_init
from src.preprocess.utils import binary_sampler, uniform_sampler, sample_batch_index


pd.options.display.max_rows = None

#  train_data= pd.read_csv('../../train_data_1.csv')

#  data_x = train_data.to_numpy()

#  gain_parameters = {
#      'batch_size': 128,
#      'hint_rate': 0.9,
#      'alpha': 100,
#      'iterations': 10000}

#train gain
#  train(data_x, gain_parameters)


#impute_data
def preprocess_4(df, norm_path, model_path):
    data_x = df.drop(['loan_id', 'user_id'], axis = 1)
    df = data_x

    data_x = data_x.to_numpy()
    imputed_data_x = impute_data(data_x, norm_path, model_path)

    imputed_data_x = pd.DataFrame(imputed_data_x)
    imputed_data_x.columns = df.columns
    #  df.update(imputed_data_x)
    return imputed_data_x



#  train_data = preprocess_4(train_data)


#  #

#  #save data
#  train_data.to_csv('../../train_data_2.csv', index = False)


#save model
#  G_sample.save('../../G_sample')
#  G_sample_str = tf.strings.format("{}", G_sample)
#  print(G_sample_str)
#  tf.io.write_file("G_sample.txt", G_sample_str)


#  # 读取文件
#  G_sample_str = tf.io.read_file("G_sample.txt")
#  # 转换回张量
#  G_sample = tf.strings.to_number(tf.strings.split(G_sample_str))
#
