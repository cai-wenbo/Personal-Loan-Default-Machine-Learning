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

    earlest_date = pd.Timestamp('2007-01-01')
    latest_date = pd.Timestamp('2019-01-01')
    #  df.update(df.loc[df['issue_date'] < earlest_date, 'issue_date'].replace(np.nan))
    #  df.update(df.loc[df['issue_date'] > latest_date, 'issue_date'].replace(np.nan))
    df.loc[:,'issue_date'] = df.loc[:,'issue_date'].apply(lambda x: x if x < latest_date else np.nan)
    df.loc[:,'issue_date'] = df.loc[:,'issue_date'].apply(lambda x: x if x > earlest_date else np.nan)

    i_y = df['issue_date'].dt.year
    i_d = df['issue_date'].dt.dayofweek
    df = df.assign(issue_year=i_y, issue_dayofweek=i_d)

    #  df.loc[:,'issue_year'] = i_y
    #  df.loc[:,'issue_dayofweek'] = i_d
    df = df.drop(['issue_date'], axis = 1)


    

    
    if 'policy_code' in df.columns:
        df = df.drop(['policy_code'], axis = 1)
    if 'work_year' in df.columns:
        wy = df['work_year'].apply(extract_number)
        df = df.drop(['work_year'], axis = 1)
        df['work_year'] = wy

    if 'house_exist' in df.columns:
        df['house_exist'] = df['house_exist'].clip(upper=2, lower = 0)
        

    if 'class' in df.columns:
        c = df['class'].apply(lambda x: ord(x))
        df = df.drop(['class'], axis = 1)
        df['class'] = c

    if 'sub_class' in df.columns:
        sc = df['sub_class'].apply(extract_number)
        df = df.drop(['sub_class'], axis = 1)
        df['sub_class'] = sc

    if 'censor_status' in df.columns:
        df = one_hot_encoding(df, 'employer_type')

    if 'employer_type' in df.columns:
        df.loc[:, 'employer_type'] = df.loc[:,'employer_type'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'employer_type')

    if 'industry' in df.columns:
        df.loc[:, 'industry'] = df.loc[:,'industry'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'industry')
        
    if 'work_type' in df.columns:
        df.loc[:, 'work_type'] = df.loc[:,'work_type'].apply(chinese_to_string)
        df = one_hot_encoding(df, 'work_type')

    if 'house_loan_status' in df.columns:
        df = one_hot_encoding(df, 'house_loan_status')

    if 'offsprings' in df.columns:
        o = df['offsprings'].apply(lambda x: 1 if x > 0 else x)
        df['havent_springs'] = o
        df.loc[:,'offsprings'] = df.loc[:,'offsprings'].apply(lambda x: x if x > 0 else np.nan)

    if 'known_dero' in df.columns:
        df['known_dero'] = df['known_dero'].clip(upper=2, lower = 0)

    if 'pub_dero_bankrup' in df.columns:
        df['pub_dero_bankrup'] = df['pub_dero_bankrup'].clip(upper=2, lower = 0)

    if 'recircle_b' in df.columns:
        df['recircle_b'] = df['recircle_b'].apply(lambda x: x if x > 0 else np.nan)
        
    if 'recircle_u' in df.columns:
        df['recircle_u'] = df['recircle_u'].apply(lambda x: x if x > 0 else np.nan)

    if 'f0' in df.columns:
        df['f0'] = df['f0'].clip(upper=30, lower = 0)

    if 'f1' in df.columns:
        df['f1'] = df['f1'].clip(upper=1, lower = 0)

    if 'f2' in df.columns:
        df['f2'] = df['f2'].clip(upper=50, lower = 0)

    if 'f3' in df.columns:
        df['f3'] = df['f3'].clip(upper=50, lower = 0)

    if 'f3' in df.columns:
        df['f3'] = df['f3'].clip(upper=30, lower = 0)

    if 'f5' in df.columns:
        df['f5'] = df['f5'].clip(upper=8, lower = 0)

    if 'early_return_amount' in df.columns:
        df['early_return_amount'] = df['early_return_amount'].clip(upper=20000, lower = 0)

    if 'early_return_amount_3mon' in df.columns:
        df['early_return_amount_3mon'] = df['early_return_amount_3mon'].clip(upper = 5000, lower = 0)
    
    if 'upper_total_loan' in df.columns:
        upper_total_loan = 50000
        #  lower_total_loan= 50000
        df.loc[:,'total_loan'] = df.loc[:,'total_loan'].apply(lambda x: x if x < upper_total_loan*10 else np.nan)
        df.loc[:,'total_loan'] = df.loc[:,'total_loan'].apply(lambda x: x if x < upper_total_loan else upper_total_loan)
        df.loc[:,'total_loan'] = df.loc[:,'total_loan'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'total_loan'] = df.loc[:,'total_loan'].apply(lambda x: x if x > lower_total_loan else np.nan)


    if 'interest' in df.columns:
        upper_interest = 50
        #  lower_interest= 3
        df.loc[:,'interest'] = df.loc[:,'interest'].apply(lambda x: x if x < upper_interest*10 else np.nan)
        df.loc[:,'interest'] = df.loc[:,'interest'].apply(lambda x: x if x < upper_interest else upper_interest)
        df.loc[:,'interest'] = df.loc[:,'interest'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'interest'] = df.loc[:,'interest'].apply(lambda x: x if x > lower_interest else np.nan)

    if 'monthly_payment' in df.columns:
        upper_monthly_payment = 2000 
        df.loc[:,'monthly_payment'] = df.loc[:,'monthly_payment'].apply(lambda x: x if x < upper_monthly_payment*10 else np.nan)
        df.loc[:,'monthly_payment'] = df.loc[:,'monthly_payment'].apply(lambda x: x if x < upper_monthly_payment else upper_monthly_payment)
        df.loc[:,'monthly_payment'] = df.loc[:,'monthly_payment'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'monthly_payment'] = df.loc[:,'monthly_payment'].apply(lambda x: x if x > lower_monthly_payment else np.nan)



    if 'debt_loan_ratio' in df.columns:
        upper_radio = 100
        lower_radio = 4
        df.loc[:,'debt_loan_ratio'] = df.loc[:,'debt_loan_ratio'].apply(lambda x: x if x < upper_radio else np.nan)
        df.loc[:,'debt_loan_ratio'] = df.loc[:,'debt_loan_ratio'].apply(lambda x: x if x > 0 else np.nan)
        df.loc[:,'debt_loan_ratio'] = df.loc[:,'debt_loan_ratio'].apply(lambda x: x if x > lower_radio else np.nan)
    
    
    if 'del_in_18month' in df.columns:
        upper_del_in_18month = 20 
        #  lower_del_in_18month= 3
        df.loc[:,'del_in_18month'] = df.loc[:,'del_in_18month'].apply(lambda x: x if x < upper_del_in_18month*2 else np.nan)
        df.loc[:,'del_in_18month'] = df.loc[:,'del_in_18month'].apply(lambda x: x if x < upper_del_in_18month else upper_del_in_18month)
        df.loc[:,'del_in_18month'] = df.loc[:,'del_in_18month'].apply(lambda x: x if x >= 0 else np.nan)
        #  df.loc[:,'del_in_18month'] = df.loc[:,'del_in_18month'].apply(lambda x: x if x > lower_del_in_18month else np.nan)

    if 'scoring_low' in df.columns:
        upper_scoring_low = 1000
        lower_scoring_low= 500
        df.loc[:,'scoring_low'] = df.loc[:,'scoring_low'].apply(lambda x: x if x < upper_scoring_low*2 else np.nan)
        df.loc[:,'scoring_low'] = df.loc[:,'scoring_low'].apply(lambda x: x if x < upper_scoring_low else upper_scoring_low)
        #  df.loc[:,'scoring_low'] = df.loc[:,'scoring_low'].apply(lambda x: x if x > 0 else np.nan)
        df.loc[:,'scoring_low'] = df.loc[:,'scoring_low'].apply(lambda x: x if x > lower_scoring_low else lower_scoring_low)

    if 'scoring_high' in df.columns:
        upper_scoring_high = 1500
        lower_scoring_high= 500
        df.loc[:,'scoring_high'] = df.loc[:,'scoring_high'].apply(lambda x: x if x < upper_scoring_high*2 else np.nan)
        df.loc[:,'scoring_high'] = df.loc[:,'scoring_high'].apply(lambda x: x if x < upper_scoring_high else upper_scoring_high)
        #  df.loc[:,'scoring_high'] = df.loc[:,'scoring_high'].apply(lambda x: x if x > 0 else np.nan)
        df.loc[:,'scoring_high'] = df.loc[:,'scoring_high'].apply(lambda x: x if x > lower_scoring_high else lower_scoring_high)
 
    if 'known_outstanding_loan' in df.columns:
        upper_known_outstanding_loan = 60
        #  lower_known_outstanding_loan= 3
        df.loc[:,'known_outstanding_loan'] = df.loc[:,'known_outstanding_loan'].apply(lambda x: x if x < upper_known_outstanding_loan*2 else np.nan)
        df.loc[:,'known_outstanding_loan'] = df.loc[:,'known_outstanding_loan'].apply(lambda x: x if x < upper_known_outstanding_loan else upper_known_outstanding_loan)
        df.loc[:,'known_outstanding_loan'] = df.loc[:,'known_outstanding_loan'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'known_outstanding_loan'] = df.loc[:,'known_outstanding_loan'].apply(lambda x: x if x > lower_known_outstanding_loan else np.nan)

    if 'recircle_b' in df.columns:
        upper_recircle_b = 1000000
        lower_recircle_b= 500
        df.loc[:,'recircle_b'] = df.loc[:,'recircle_b'].apply(lambda x: x if x < upper_recircle_b*2 else np.nan)
        df.loc[:,'recircle_b'] = df.loc[:,'recircle_b'].apply(lambda x: x if x < upper_recircle_b else upper_recircle_b)
        #  df.loc[:,'recircle_b'] = df.loc[:,'recircle_b'].apply(lambda x: x if x > 0 else np.nan)
        df.loc[:,'recircle_b'] = df.loc[:,'recircle_b'].apply(lambda x: x if x > lower_recircle_b else lower_recircle_b)

    if 'recircle_u' in df.columns:
        upper_recircle_u = 150 
        lower_recircle_u= 20
        df.loc[:,'recircle_u'] = df.loc[:,'recircle_u'].apply(lambda x: x if x < upper_recircle_u*2 else np.nan)
        df.loc[:,'recircle_u'] = df.loc[:,'recircle_u'].apply(lambda x: x if x < upper_recircle_u else upper_recircle_u)
        #  df.loc[:,'recircle_u'] = df.loc[:,'recircle_u'].apply(lambda x: x if x > 0 else np.nan)
        df.loc[:,'recircle_u'] = df.loc[:,'recircle_u'].apply(lambda x: x if x > lower_recircle_u else lower_recircle_u)



    #  df.update(df.loc[df['debt_loan_ratio'] == 0, 'debt_loan_ratio'].replace(np.nan))
    #  df.update(df.loc[df['debt_loan_ratio'] > upper_radio, 'debt_loan_ratio'].replace(np.nan))

    #  df['income'] = df['monthly_payment'] * df['year_of_loan'] /df['debt_loan_ratio']
    df['income'] = df.eval('monthly_payment * year_of_loan / debt_loan_ratio')



    if 'total_loan' in df.columns:
        df['total_loan'] = np.log1p(df['total_loan'])

    if 'interest' in df.columns:
        df['interest'] = np.log1p(df['interest'])
    if 'monthly_payment' in df.columns:
        df['monthly_payment'] = np.log1p(df['monthly_payment'])
    df['income'] = np.log1p(df['income'])

    if 'del_in_18month' in df.columns:
        df['del_in_18month'] = np.log1p(df['del_in_18month'])
    if 'known_outstanding_loan' in df.columns:
        df['known_outstanding_loan'] = np.log1p(df['known_outstanding_loan'])

    if 'recircle_b' in df.columns:
        df['recircle_b'] = np.log1p(df['recircle_b'])
    if 'recircle_u' in df.columns:
        df['recircle_u'] = np.log1p(df['recircle_u'])
    if 'f0' in df.columns:
        df['f0'] = np.log1p(df['f0'])
    if 'f2' in df.columns:
        df['f2'] = np.log1p(df['f2'])
    if 'f3' in df.columns:
        df['f3'] = np.log1p(df['f3'])
    if 'f4' in df.columns:
        df['f4'] = np.log1p(df['f4'])
    if 'f5' in df.columns:
        df['f5'] = np.log1p(df['f5'])
    if 'early_return_amount' in df.columns:
        df['early_return_amount'] = np.log1p(df['early_return_amount'])
    if 'early_return_amount_3mon' in df.columns:
        df['early_return_amount_3mon'] = np.log1p(df['early_return_amount_3mon'])

    df.loc[df['early_return_amount_3mon'] > 0, 'early_return'] = 1
    df.loc[df['early_return_amount'] > 0, 'early_return'] = 1
    df.loc[df['early_return_amount'] == 0, 'early_return'] = 0


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
    with open('data/analysis_data/saved_dicts.pkl', 'rb') as f:
        issue_year_proportions = pickle.load(f)
        issue_dayofweek_proportions = pickle.load(f)
        use_proportions = pickle.load(f)
        post_code_proportions = pickle.load(f)
        region_proportions = pickle.load(f)
        title_proportions = pickle.load(f)
        earlies_credit_mon_proportions = pickle.load(f)

        issue_year_dict = pickle.load(f)
        issue_dayofweek_dict = pickle.load(f)
        use_dict = pickle.load(f)
        post_code_dict = pickle.load(f)
        region_dict = pickle.load(f)
        title_dict = pickle.load(f)
        earlies_credit_mon_dict = pickle.load(f)

    df = extract_category(df, 'issue_year', issue_year_proportions, issue_year_dict, 5)
    df = extract_category(df, 'issue_dayofweek', issue_dayofweek_proportions, issue_dayofweek_dict, 4)
    df = extract_category(df, 'use', use_proportions, use_dict, 5)
    df = extract_category(df, 'post_code', post_code_proportions, post_code_dict, 5)
    df = extract_category(df, 'region', region_proportions, region_dict, 5)
    df = extract_category(df, 'title', title_proportions, title_dict, 5)
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
    with open('data/analysis_data/cols.txt', 'r') as f:
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
