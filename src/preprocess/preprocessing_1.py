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



import pandas as pd
import numpy as np
pd.options.display.max_rows = None
#
#  #  load data
#  train1 = pd.read_csv('../../input/train_public.csv')
#  train2 = pd.read_csv('../../input/train_internet.csv')
#
#  train1['issue_date'] = pd.to_datetime(train1['issue_date'], format='%Y/%m/%d')
#  train2['issue_date'] = pd.to_datetime(train2['issue_date'], format='%Y-%m-%d')
#
#  data = pd.concat([train1, train2], ignore_index=True)
#  data['issue_date'] = data['issue_date'].astype(int) // 86400000000000
#
#
#  #  get a smaller dataset to test
#  proportion = 0.01
#  sample_size = int(len(data) * proportion)
#  data = data.sample(frac = proportion)



#  preprocesssing

#  total_loan
#  year_of_loan
#  interest
#  monthly_payment
#  class
#  sub_class
#  issue_date
#  work_type
#  employer_type
#  industry
#  work_year
#  house_exist
#  censor_status
#  use
#  post_code
#  region
#  title
#  offsprings
#  marriage
#  debt_loan_ratio
#  del_in_18month
#  scoring_low
#  scoring_high
#  known_outstanding_loan
#  known_dero
#  pub_dero_bankrup
#  recircle_b
#  recircle_u
#  initial_list_status
#  app_type
#  policy_code
#  f0
#  f1
#  f2
#  f3
#  f4
#  f5
#  early_return
#  early_return_amount
#  early_return_amount_3mon
#  house_loan_status

#  is_default


#  #split to train_data and test_data
#
#  test_proportion = 0.1
#  test_size = int(len(data) * test_proportion)
#
#  test_indices = np.random.choice(data.index, size=test_size, replace=False)
#
#  train_data = data[~data.index.isin(test_indices)]
#  test_data = data[data.index.isin(test_indices)]


#general preprocessing

import re
def extract_number(x):
    if not isinstance(x, str):
        return x
    numbers = re.findall(r'\d+', x)
    if len(numbers) > 0:
        return float(''.join(numbers))
    else:
        return np.nan

def chinese_to_string(chinese_str):
    if  isinstance(chinese_str, float):
        return "NaN";
    else:
        s = "" 
        char1 = chr(ord('a') + int(str(ord(chinese_str[0]))[0]))
        char2 = chr(ord('a') + int(str(ord(chinese_str[0]))[1]))
        char3 = chr(ord('a') + int(str(ord(chinese_str[1]))[0]))
        char4 = chr(ord('a') + int(str(ord(chinese_str[1]))[1]))
        s = s + char1 + char2 + char3 + char4
        return s 

def one_hot_encoding(df, col):
    one_hot_col = pd.get_dummies(df[col], prefix=col, dtype = int)
    df = df.drop([col], axis = 1)
    df = pd.concat([df, one_hot_col], axis = 1)
    return df
    

def preprocess_1(df):
    #  df.loc[:,'work_year'] = df.loc[:,'work_year'].apply(extract_number)
    #  df.loc[:,'class'] = df.loc[:,'class'].apply(lambda x: ord(x))
    #  df.loc[:,'sub_class'] = df.loc[:,'sub_class'].apply(extract_number)
    #  df['work_year'] = df.loc[:,'work_year'].apply(extract_number)
    #  df['class'] = df.loc[:,'class'].apply(lambda x: ord(x))
    #  df['sub_class'] = df.loc[:,'sub_class'].apply(extract_number)
    
    if 'policy_code' in df.columns:
        df = df.drop(['policy_code'], axis = 1)
    if 'work_year' in df.columns:
        wy = df['work_year'].apply(extract_number)
        df = df.drop(['work_year'], axis = 1)
        df['work_year'] = wy

    if 'class' in df.columns:
        c = df['class'].apply(lambda x: ord(x))
        df = df.drop(['class'], axis = 1)
        df['class'] = c

    if 'sub_class' in df.columns:
        sc = df['sub_class'].apply(extract_number)
        df = df.drop(['sub_class'], axis = 1)
        df['sub_class'] = sc

    if 'employer_type' in df.columns:
        df.loc[:, 'employer_type'] = df.loc[:,'employer_type'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'employer_type')

    if 'industry' in df.columns:
        df.loc[:, 'industry'] = df.loc[:,'industry'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'industry')
        
    if 'work_type' in df.columns:
        df.loc[:, 'work_type'] = df.loc[:,'work_type'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'work_type')

    #  if 'use' in df.columns:
    #      df = one_hot_encoding(df, 'use')
    return df


#  train_data = preprocess_1(train_data)




#use train data to decide how build embedding mode
#  and embed train data
def get_category_proportions(df, col):
    value_counts = df[col].value_counts()
    proportions = value_counts / len(df)
    col_proportions = dict(zip(proportions.index, proportions.values))
    return col_proportions

def get_category_dict(df, category_col, label_col):
    cat_label_counts = df[df[label_col] == 1][category_col].value_counts()
    cat_counts = df[category_col].value_counts()
    cat_probs = {}
    for cat in cat_counts.index:
        if cat in cat_label_counts.index:
            cat_probs[cat] = cat_label_counts[cat] / cat_counts[cat] + 0.5
        else:
            cat_probs[cat] = 0.0
    return cat_probs

def extract_category(df, col, col_proportions, col_dict, n):
    p = 1.0 / n

    cols = []
    current_categories = []
    current_col_proportion = 0.0
    remaining_proportion = 1.0

    for category in col_proportions:
        proportion = col_proportions[category]
        current_categories.append(category)
        current_col_proportion += proportion
        if (current_col_proportion >= p):
            cols.append(list(current_categories))
            remaining_proportion -= current_col_proportion
            current_categories = []
            current_col_proportion = 0.0
                
    cols.append(list(current_categories))
            
    new_df = pd.DataFrame()
    for i, categories in enumerate(cols):
        new_df[col + '_' + str(i+1)] = df[col].apply(lambda x: col_dict[x] if x in categories else 0)
        
    df = pd.concat([df, new_df], axis= 1)
    df = df.drop([col], axis = 1)

    return df

import pickle
def preprocess_2(df):
    with open('saved_dicts.pkl', 'rb') as f:
        use_proportions = pickle.load(f)
        post_code_proportions = pickle.load(f)
        region_proportions = pickle.load(f)
        title_proportions = pickle.load(f)
        earlies_credit_mon_proportions = pickle.load(f)

        use_dict = pickle.load(f)
        post_code_dict = pickle.load(f)
        region_dict = pickle.load(f)
        title_dict = pickle.load(f)
        earlies_credit_mon_dict = pickle.load(f)

    df = extract_category(df, 'use', use_proportions, use_dict, 10)
    df = extract_category(df, 'post_code', post_code_proportions, post_code_dict, 10)
    df = extract_category(df, 'region', region_proportions, region_dict, 10)
    df = extract_category(df, 'title', title_proportions, title_dict, 10)
    df = extract_category(df, 'earlies_credit_mon', earlies_credit_mon_proportions, earlies_credit_mon_dict, 4)

    

    return df








#  post_code_proportions = get_category_proportions(train_data, 'post_code')
#  region_proportions = get_category_proportions(train_data, 'region')
#  title_proportions = get_category_proportions(train_data, 'title')
#  earlies_credit_mon_proportions = get_category_proportions(train_data, 'earlies_credit_mon')
#
#  post_code_dict = get_category_dict(train_data, 'post_code', 'is_default')
#  region_dict = get_category_dict(train_data, 'region', 'is_default')
#  title_dict = get_category_dict(train_data, 'title', 'is_default')
#  earlies_credit_mon_dict = get_category_dict(train_data, 'earlies_credit_mon', 'is_default')

#  save dict

#  import pickle
#  with open('../../saved_dicts.pkl', 'wb') as f:
#      pickle.dump(post_code_proportions, f)
#      pickle.dump(region_proportions, f)
#      pickle.dump(title_proportions, f)
#      pickle.dump(earlies_credit_mon_proportions, f)
#
#      pickle.dump(post_code_dict, f)
#      pickle.dump(region_dict, f)
#      pickle.dump(title_dict, f)
#      pickle.dump(earlies_credit_mon_dict, f)


#  train_data = preprocess_2(train_data)


#test

#  print(train_data.dtypes)
#  print("title_5")
#  print(train_data['title_5'].unique())

#      use train data to train GAIN


#  # 把多个dict顺序地从文件加载
#  with open('saved_dicts.pkl', 'rb') as f:
#      loaded_dict1 = pickle.load(f)
#      loaded_dict2 = pickle.load(f)
#      loaded_dict3 = pickle.load(f)

#  #  save data
#  train_data.to_csv('../../train_data_1.csv', index = False)
#  test_data.to_csv('../../test_data_1.csv', index = False)


#make df1 and df2 have the same columns

#  def preprocess_3(df1, df2):
#      for col in df2.columns:
#          if col not in df1.columns:
#              df1[col] = df2[col]
#      for col in df1.columns:
#          if col not in df2.columns:
#              df1.drop(col, axis=1, inplace=True)
#      return df1

def preprocess_3(df):
    with open('cols.txt', 'r') as f:
        cols = f.read().strip().split(',')

    #  add
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    #  delete
    df = df[cols]
    
    df = df.reindex(columns=cols)
    df = pd.DataFrame(df)
    return df
