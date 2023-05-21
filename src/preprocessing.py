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
from pandas._libs.tslibs.period import from_ordinals
from pandas.core.indexes.base import default_pprint
pd.options.display.max_rows = None



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

def one_hot_encoding(df1, col):
    df = df1.copy()
    one_hot_col = pd.get_dummies(df[col], prefix=col, dtype = int)
    df = df.drop([col], axis = 1)
    df = pd.concat([df, one_hot_col], axis = 1)
    return df
    







#use train data to decide how build embedding mode
#  and embed train data
from collections import OrderedDict

def get_category_proportions(df, col):
    value_counts = df[col].value_counts()
    proportions = value_counts / len(df)
    col_proportions = dict(zip(proportions.index, proportions.values))
    #  col_proportions = OrderedDict(col_proportions)
    col_proportions = OrderedDict(sorted(col_proportions.items(), key=lambda x: x[1], reverse=True))

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

    cat_probs = OrderedDict(sorted(cat_probs.items(), key=lambda x: x[1], reverse=True))
    return cat_probs

def get_embedding_dicts(df1, embedding_path):
    df = df1.copy()
    post_code_proportions = get_category_proportions(df, 'post_code')
    region_proportions = get_category_proportions(df, 'region')
    title_proportions = get_category_proportions(df, 'title')
    #  earlies_credit_mon_proportions = get_category_proportions(df, 'earlies_credit_mon')

    #  issue_year_dict = get_category_dict(df, 'issue_year', 'is_default')
    #  issue_dayofweek_dict = get_category_dict(df, 'issue_dayofweek', 'is_default')
    #  use_dict = get_category_dict(df, 'use', 'is_default')
    post_code_dict = get_category_dict(df, 'post_code', 'is_default')
    region_dict = get_category_dict(df, 'region', 'is_default')
    title_dict = get_category_dict(df, 'title', 'is_default')
    #  earlies_credit_mon_dict = get_category_dict(df, 'earlies_credit_mon', 'is_default')

    #save dict and proportions
    import pickle

    with open(embedding_path, 'wb') as f:
        #  pickle.dump(issue_year_proportions, f)
        #  pickle.dump(issue_dayofweek_proportions, f)
        #  pickle.dump(use_proportions, f)
        pickle.dump(post_code_proportions, f)
        pickle.dump(region_proportions, f)
        pickle.dump(title_proportions, f)
        #  pickle.dump(earlies_credit_mon_proportions, f)

        #  pickle.dump(issue_year_dict, f)
        #  pickle.dump(issue_dayofweek_dict, f)
        #  pickle.dump(use_dict, f)
        pickle.dump(post_code_dict, f)
        pickle.dump(region_dict, f)
        pickle.dump(title_dict, f)
        #  pickle.dump(earlies_credit_mon_dict, f)


def extract_category(df1, col, col_proportions, col_dict, n):
    #  print(col_proportions)
    #  print(col_dict)
    df = df1.copy()
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

def convert(df1):
    df = df1.copy()

    if 'policy_code' in df.columns:
        df = df.drop(['policy_code'], axis = 1)
    if 'loan_id' in df.columns:
        df = df.drop(['loan_id'], axis = 1)
    if 'user_id' in df.columns:
        df = df.drop(['user_id'], axis = 1)

    if 'earlies_credit_mon' in df.columns:
        df['earlies_credit_month'] = df['earlies_credit_mon'].apply(lambda x: re.findall('[a-zA-Z]+', x)[0])
        #  print(df['earlies_credit_month'].value_counts())
        month_mapping = {
            'Jan': 0,
            'Feb': 1,
            'Mar': 2,
            'Apr': 3,
            'May': 4,
            'Jun': 5,
            'Jul': 6,
            'Aug': 7,
            'Sep': 8,
            'Oct': 9,
            'Nov': 10,
            'Dec': 11
        }
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


        #  print(df['earlies_credit_month'].value_counts())
        df['earlies_credit_month'] = df['earlies_credit_month'].apply(lambda x: month_mapping[x] if x in months else 0)
        #  print(df['earlies_credit_month'].value_counts())


        pattern_year = r'([a-zA-Z]+)-(\d+)'
        df['earlies_credit_year'] = df['earlies_credit_mon'].str.extract(pattern_year, expand=False)[1]
        df['earlies_credit_year'] = pd.to_numeric(df['earlies_credit_year'], errors='coerce')

        df['earlies_credit_year'] = df['earlies_credit_year'].apply(lambda x: x + 100 if x < 20 else x)
        df['earlies_credit_year'] = df['earlies_credit_year'].apply(lambda x: x - 1900 if x > 1900 else x)

        df['earlies_credit_mon'] = df.eval('earlies_credit_year * 12 + earlies_credit_month')

        df = df.drop(['earlies_credit_month', 'earlies_credit_year'], axis = 1)


    if 'issue_date' in df.columns:
        earlest_date = pd.Timestamp('2007-01-01')
        latest_date = pd.Timestamp('2019-01-01')
        df['issue_date'] = df['issue_date'].apply(lambda x: x if x < latest_date else np.nan)
        df['issue_date'] = df['issue_date'].apply(lambda x: x if x > earlest_date else np.nan)

        df['issue_year'] = df['issue_date'].dt.year
        df['issue_dayofweek'] = df['issue_date'].dt.dayofweek

        df = df.drop(['issue_date'], axis = 1)

    industry_dict = {'金融业': 0, '公共服务、社会组织': 1, '文化和体育业': 2, '信息传输、软件和信息技术服务业': 3, '制造业': 4, '住宿和餐饮业': 5, '建筑业': 6, '电力、热力生产供应业': 7, '房地产业': 8, '交通运输、仓储和邮政业': 9, '批发和零售业': 10, '农、林、牧、渔业': 11, '采矿业': 12, '国际组织': 13}
    industry_cats = ['金融业', '公共服务、社会组织', '文化和体育业', '信息传输、软件和信息技术服务业', '制造业', '住宿和餐饮业', '建筑业', '电力、热力生产供应业', '房地产业', '交通运输、仓储和邮政业', '批发和零售业', '农、林、牧、渔业', '采矿业', '国际组织']

    employer_type_dict = {'政府机构': 0, '世界五百强': 1, '幼教与中小学校': 2, '高等教育机构': 3, '普通企业': 4, '上市企业': 5}
    employer_type_cats = ['政府机构', '世界五百强', '幼教与中小学校', '高等教育机构', '普通企业', '上市企业']

    work_type_dict = {'职员': 0, '其他': 1, '工人': 2, '工程师': 3, '公务员': 4}
    work_type_cats = ['职员', '其他', '工人', '工程师', '公务员']


    if 'employer_type' in df.columns:
        df['employer_type'] = df['employer_type'].apply(lambda x: employer_type_dict[x] if x in employer_type_cats else -1)

    if 'industry' in df.columns:
        df['industry'] = df['industry'].apply(lambda x: industry_dict[x] if x in industry_cats else -1)
        
    if 'work_type' in df.columns:
        df['work_type'] = df['work_type'].apply(lambda x: work_type_dict[x] if x in work_type_cats else -1)

    if 'work_year' in df.columns:
        df['work_over_10ys'] = np.where(df['work_year'].str.contains('\+'), 1, np.where(df['work_year'].notnull(), 0, np.nan))

    if 'work_year' in df.columns:
        df['work_year'] = df['work_year'].apply(extract_number)

    if 'class' in df.columns:
        df['class'] = df['class'].apply(lambda x: ord(x))

    if 'sub_class' in df.columns:
        df['sub_class'] = df['sub_class'].apply(extract_number)


    return df


def clean(df1):
    df = df1.copy()

    
    if 'upper_total_loan' in df.columns:
        upper_total_loan = 50000
        #  lower_total_loan= 50000
        df['total_loan'] = df['total_loan'].apply(lambda x: x if x < upper_total_loan*10 else np.nan)
        df['total_loan'] = df['total_loan'].apply(lambda x: x if x < upper_total_loan else upper_total_loan)
        df['total_loan'] = df['total_loan'].apply(lambda x: x if x > 0 else np.nan)

    if 'upper_year_of_loan' in df.columns:
        upper_year_of_loan = 5
        #  lower_year_of_loan= 50000
        df['year_of_loan'] = df['year_of_loan'].apply(lambda x: x if x < upper_year_of_loan*10 else np.nan)
        df['year_of_loan'] = df['year_of_loan'].apply(lambda x: x if x < upper_year_of_loan else upper_year_of_loan)
        df['year_of_loan'] = df['year_of_loan'].apply(lambda x: x if x > 0 else np.nan)

    if 'interest' in df.columns:
        upper_interest = 50
        #  lower_interest= 3
        df['interest'] = df['interest'].apply(lambda x: x if x < upper_interest*10 else np.nan)
        df['interest'] = df['interest'].apply(lambda x: x if x < upper_interest else upper_interest)
        df['interest'] = df['interest'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'interest'] = df.loc[:,'interest'].apply(lambda x: x if x > lower_interest else np.nan)

    if 'monthly_payment' in df.columns:
        upper_monthly_payment = 2000 
        df['monthly_payment'] = df['monthly_payment'].apply(lambda x: x if x < upper_monthly_payment*10 else np.nan)
        df['monthly_payment'] = df['monthly_payment'].apply(lambda x: x if x < upper_monthly_payment else upper_monthly_payment)
        df['monthly_payment'] = df['monthly_payment'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'monthly_payment'] = df.loc[:,'monthly_payment'].apply(lambda x: x if x > lower_monthly_payment else np.nan)

    if 'upper_class' in df.columns:
        upper_class = 71
        lower_class= 65
        df['class'] = df['class'].apply(lambda x: x if x <= upper_class else np.nan)
        df['class'] = df['class'].apply(lambda x: x if x >= lower_class else np.nan)

    if 'upper_sub_class' in df.columns:
        upper_sub_class = 5
        lower_sub_class= 1 
        df['sub_class'] = df['sub_class'].apply(lambda x: x if x <= upper_sub_class else np.nan)
        df['sub_class'] = df['sub_class'].apply(lambda x: x if x >= lower_sub_class else np.nan)

    if 'upper_work_year' in df.columns:
        upper_work_year = 10
        lower_work_year= 0 
        df['work_year'] = df['work_year'].apply(lambda x: x if x <= upper_work_year else upper_work_year)
        df['work_year'] = df['work_year'].apply(lambda x: x if x >= lower_work_year else np.nan)

    if 'upper_house_exist' in df.columns:
        upper_house_exist = 5
        lower_house_exist= 0
        df['house_exist'] = df['house_exist'].apply(lambda x: x if x <= upper_house_exist else upper_house_exist)
        df['house_exist'] = df['house_exist'].apply(lambda x: x if x >= lower_house_exist else np.nan)

    if 'upper_censor_status' in df.columns:
        upper_censor_status = 2
        lower_censor_status= 0
        df['censor_status'] = df['censor_status'].apply(lambda x: x if x <= upper_censor_status else np.nan)
        df['censor_status'] = df['censor_status'].apply(lambda x: x if x >= lower_censor_status else np.nan)

    if 'upper_use' in df.columns:
        upper_use = 14
        lower_use= 0
        #  df['use'] = df['use'].apply(lambda x: x if x < upper_use*10 else np.nan)
        df['use'] = df['use'].apply(lambda x: x if x <= upper_use else np.nan)
        df['use'] = df['use'].apply(lambda x: x if x >= lower_use else np.nan)

    if 'upper_region' in df.columns:
        upper_region = 50
        lower_region= 1
        #  df['region'] = df['region'].apply(lambda x: x if x < upper_region*10 else np.nan)
        df['region'] = df['region'].apply(lambda x: x if x <= upper_region else np.nan)
        df['region'] = df['region'].apply(lambda x: x if x >= lower_region else np.nan)

    if 'upper_offsprings' in df.columns:
        upper_offsprings = 5
        lower_offsprings= 0
        #  df['offsprings'] = df['offsprings'].apply(lambda x: x if x < upper_offsprings*10 else np.nan)
        df['offsprings'] = df['offsprings'].apply(lambda x: x if x <= upper_offsprings else np.nan)
        df['offsprings'] = df['offsprings'].apply(lambda x: x if x >= lower_offsprings else np.nan)
    
    if 'upper_marriage' in df.columns:
        upper_marriage = 3
        lower_marriage= 0
        #  df['marriage'] = df['marriage'].apply(lambda x: x if x < upper_marriage*10 else np.nan)
        df['marriage'] = df['marriage'].apply(lambda x: x if x <= upper_marriage else upper_marriage)
        df['marriage'] = df['marriage'].apply(lambda x: x if x >= lower_marriage else np.nan)

    if 'debt_loan_ratio' in df.columns:
        upper_radio = 100
        lower_radio = 4
        df['debt_loan_ratio'] = df['debt_loan_ratio'].apply(lambda x: x if x < upper_radio*10 else upper_radio)
        df['debt_loan_ratio'] = df['debt_loan_ratio'].apply(lambda x: x if x < upper_radio else upper_radio)
        df['debt_loan_ratio'] = df['debt_loan_ratio'].apply(lambda x: x if x > 0 else np.nan)
        df['debt_loan_ratio'] = df['debt_loan_ratio'].apply(lambda x: x if x > lower_radio else lower_radio)    
    
    if 'del_in_18month' in df.columns:
        upper_del_in_18month = 20 
        #  lower_del_in_18month= 3
        df['del_in_18month'] = df['del_in_18month'].apply(lambda x: x if x < upper_del_in_18month*2 else np.nan)
        df['del_in_18month'] = df['del_in_18month'].apply(lambda x: x if x < upper_del_in_18month else upper_del_in_18month)
        df['del_in_18month'] = df['del_in_18month'].apply(lambda x: x if x >= 0 else np.nan)
        #  df.loc[:,'del_in_18month'] = df.loc[:,'del_in_18month'].apply(lambda x: x if x > lower_del_in_18month else np.nan)

    if 'scoring_low' in df.columns:
        upper_scoring_low = 1000
        lower_scoring_low= 500
        df['scoring_low'] = df['scoring_low'].apply(lambda x: x if x < upper_scoring_low*2 else np.nan)
        df['scoring_low'] = df['scoring_low'].apply(lambda x: x if x < upper_scoring_low else upper_scoring_low)
        #  df.loc[:,'scoring_low'] = df.loc[:,'scoring_low'].apply(lambda x: x if x > 0 else np.nan)
        df['scoring_low'] = df['scoring_low'].apply(lambda x: x if x > lower_scoring_low else lower_scoring_low)

    if 'scoring_high' in df.columns:
        upper_scoring_high = 1500
        lower_scoring_high= 500
        df['scoring_high'] = df['scoring_high'].apply(lambda x: x if x < upper_scoring_high*2 else np.nan)
        df['scoring_high'] = df['scoring_high'].apply(lambda x: x if x < upper_scoring_high else upper_scoring_high)
        #  df.loc[:,'scoring_high'] = df.loc[:,'scoring_high'].apply(lambda x: x if x > 0 else np.nan)
        df['scoring_high'] = df['scoring_high'].apply(lambda x: x if x > lower_scoring_high else lower_scoring_high)
 
    if 'known_outstanding_loan' in df.columns:
        upper_known_outstanding_loan = 60
        #  lower_known_outstanding_loan= 3
        df['known_outstanding_loan'] = df['known_outstanding_loan'].apply(lambda x: x if x < upper_known_outstanding_loan*2 else np.nan)
        df['known_outstanding_loan'] = df['known_outstanding_loan'].apply(lambda x: x if x < upper_known_outstanding_loan else upper_known_outstanding_loan)
        df['known_outstanding_loan'] = df['known_outstanding_loan'].apply(lambda x: x if x > 0 else np.nan)
        #  df.loc[:,'known_outstanding_loan'] = df.loc[:,'known_outstanding_loan'].apply(lambda x: x if x > lower_known_outstanding_loan else np.nan)


    if 'known_dero' in df.columns:
        upper_known_dero = 12
        lower_known_dero= 0
        #  df['known_dero'] = df['known_dero'].apply(lambda x: x if x < upper_known_dero*2 else np.nan)
        df['known_dero'] = df['known_dero'].apply(lambda x: x if x < upper_known_dero else upper_known_dero)
        #  df.loc[:,'known_dero'] = df.loc[:,'known_dero'].apply(lambda x: x if x > 0 else np.nan)
        df['known_dero'] = df['known_dero'].apply(lambda x: x if x >= lower_known_dero else lower_known_dero)

    if 'pub_dero_bankrup' in df.columns:
        upper_pub_dero_bankrup = 5
        lower_pub_dero_bankrup= 0
        #  df['pub_dero_bankrup'] = df['pub_dero_bankrup'].apply(lambda x: x if x < upper_pub_dero_bankrup*2 else np.nan)
        df['pub_dero_bankrup'] = df['pub_dero_bankrup'].apply(lambda x: x if x < upper_pub_dero_bankrup else upper_pub_dero_bankrup)
        #  df.loc[:,'pub_dero_bankrup'] = df.loc[:,'pub_dero_bankrup'].apply(lambda x: x if x > 0 else np.nan)
        df['pub_dero_bankrup'] = df['pub_dero_bankrup'].apply(lambda x: x if x >= lower_pub_dero_bankrup else lower_pub_dero_bankrup)


    if 'recircle_b' in df.columns:
        upper_recircle_b = 1000000
        lower_recircle_b= 500
        df['recircle_b'] = df['recircle_b'].apply(lambda x: x if x < upper_recircle_b*2 else np.nan)
        df['recircle_b'] = df['recircle_b'].apply(lambda x: x if x < upper_recircle_b else upper_recircle_b)
        #  df.loc[:,'recircle_b'] = df.loc[:,'recircle_b'].apply(lambda x: x if x > 0 else np.nan)
        df['recircle_b'] = df['recircle_b'].apply(lambda x: x if x > lower_recircle_b else lower_recircle_b)

    if 'recircle_u' in df.columns:
        upper_recircle_u = 150 
        lower_recircle_u= 20
        df['recircle_u'] = df['recircle_u'].apply(lambda x: x if x < upper_recircle_u*2 else np.nan)
        df['recircle_u'] = df['recircle_u'].apply(lambda x: x if x < upper_recircle_u else upper_recircle_u)
        #  df.loc[:,'recircle_u'] = df.loc[:,'recircle_u'].apply(lambda x: x if x > 0 else np.nan)
        df['recircle_u'] = df['recircle_u'].apply(lambda x: x if x > lower_recircle_u else lower_recircle_u)

    if 'initial_list_status' in df.columns:
        upper_initial_list_status = 1
        lower_initial_list_status= 0
        #  df['initial_list_status'] = df['initial_list_status'].apply(lambda x: x if x < upper_initial_list_status*2 else np.nan)
        df['initial_list_status'] = df['initial_list_status'].apply(lambda x: x if x <= upper_initial_list_status else np.nan)
        #  df.loc[:,'initial_list_status'] = df.loc[:,'initial_list_status'].apply(lambda x: x if x > 0 else np.nan)
        df['initial_list_status'] = df['initial_list_status'].apply(lambda x: x if x >= lower_initial_list_status else np.nan)
    
    if 'app_type' in df.columns:
        upper_app_type = 1
        lower_app_type= 0
        #  df['app_type'] = df['app_type'].apply(lambda x: x if x < upper_app_type*2 else np.nan)
        df['app_type'] = df['app_type'].apply(lambda x: x if x <= upper_app_type else np.nan)
        #  df.loc[:,'app_type'] = df.loc[:,'app_type'].apply(lambda x: x if x > 0 else np.nan)
        df['app_type'] = df['app_type'].apply(lambda x: x if x >= lower_app_type else np.nan)

    if 'f0' in df.columns:
        upper_f0 = 50
        lower_f0= 0
        df['f0'] = df['f0'].apply(lambda x: x if x < upper_f0*3 else np.nan)
        df['f0'] = df['f0'].apply(lambda x: x if x <= upper_f0 else upper_f0)
        #  df.loc[:,'f0'] = df.loc[:,'f0'].apply(lambda x: x if x > 0 else np.nan)
        df['f0'] = df['f0'].apply(lambda x: x if x >= lower_f0 else np.nan)

    if 'f1' in df.columns:
        upper_f1 = 3
        lower_f1= 0
        df['f1'] = df['f1'].apply(lambda x: x if x < upper_f1*3 else np.nan)
        df['f1'] = df['f1'].apply(lambda x: x if x <= upper_f1 else upper_f1)
        #  df.loc[:,'f1'] = df.loc[:,'f1'].apply(lambda x: x if x > 0 else np.nan)
        df['f1'] = df['f1'].apply(lambda x: x if x >= lower_f1 else np.nan)

    if 'f2' in df.columns:
        upper_f2 = 80
        lower_f2= 0
        df['f2'] = df['f2'].apply(lambda x: x if x < upper_f2*3 else np.nan)
        df['f2'] = df['f2'].apply(lambda x: x if x <= upper_f2 else upper_f2)
        #  df.loc[:,'f2'] = df.loc[:,'f2'].apply(lambda x: x if x > 0 else np.nan)
        df['f2'] = df['f2'].apply(lambda x: x if x >= lower_f2 else np.nan)

    if 'f3' in df.columns:
        upper_f3 = 50
        lower_f3= 0
        df['f3'] = df['f3'].apply(lambda x: x if x < upper_f3*3 else np.nan)
        df['f3'] = df['f3'].apply(lambda x: x if x <= upper_f3 else upper_f3)
        #  df.loc[:,'f3'] = df.loc[:,'f3'].apply(lambda x: x if x > 0 else np.nan)
        df['f3'] = df['f3'].apply(lambda x: x if x >= lower_f3 else np.nan)

    if 'f4' in df.columns:
        upper_f4 = 50
        lower_f4= 0
        df['f4'] = df['f4'].apply(lambda x: x if x < upper_f4*3 else np.nan)
        df['f4'] = df['f4'].apply(lambda x: x if x <= upper_f4 else upper_f4)
        #  df.loc[:,'f4'] = df.loc[:,'f4'].apply(lambda x: x if x > 0 else np.nan)
        df['f4'] = df['f4'].apply(lambda x: x if x >= lower_f4 else np.nan)

    if 'f5' in df.columns:
        upper_f5 = 20
        lower_f5= 0
        df['f5'] = df['f5'].apply(lambda x: x if x < upper_f5*3 else np.nan)
        df['f5'] = df['f5'].apply(lambda x: x if x <= upper_f5 else upper_f5)
        #  df.loc[:,'f5'] = df.loc[:,'f5'].apply(lambda x: x if x > 0 else np.nan)
        df['f5'] = df['f5'].apply(lambda x: x if x >= lower_f5 else np.nan)

    if 'early_return' in df.columns:
        upper_early_return = 1
        lower_early_return= 0
        #  df['early_return'] = df['early_return'].apply(lambda x: x if x < upper_early_return*2 else np.nan)
        df['early_return'] = df['early_return'].apply(lambda x: x if x <= upper_early_return else np.nan)
        #  df.loc[:,'early_return'] = df.loc[:,'early_return'].apply(lambda x: x if x > 0 else np.nan)
        df['early_return'] = df['early_return'].apply(lambda x: x if x >= lower_early_return else np.nan)

    if 'early_return_amount' in df.columns:
        upper_early_return_amount = 20000
        lower_early_return_amount= 0
        df['early_return_amount'] = df['early_return_amount'].apply(lambda x: x if x < upper_early_return_amount*10 else np.nan)
        df['early_return_amount'] = df['early_return_amount'].apply(lambda x: x if x <= upper_early_return_amount else upper_early_return_amount)
        #  df.loc[:,'early_return_amount'] = df.loc[:,'early_return_amount'].apply(lambda x: x if x > 0 else np.nan)
        df['early_return_amount'] = df['early_return_amount'].apply(lambda x: x if x > lower_early_return_amount else np.nan)

    if 'early_return_amount_3mon' in df.columns:
        upper_early_return_amount_3mon = 5000
        lower_early_return_amount_3mon= 0
        df['early_return_amount_3mon'] = df['early_return_amount_3mon'].apply(lambda x: x if x < upper_early_return_amount_3mon*10 else np.nan)
        df['early_return_amount_3mon'] = df['early_return_amount_3mon'].apply(lambda x: x if x <= upper_early_return_amount_3mon else upper_early_return_amount_3mon)
        #  df.loc[:,'early_return_amount_3mon'] = df.loc[:,'early_return_amount_3mon'].apply(lambda x: x if x > 0 else np.nan)
        df['early_return_amount_3mon'] = df['early_return_amount_3mon'].apply(lambda x: x if x > lower_early_return_amount_3mon else np.nan)

    df.loc[df['early_return_amount_3mon'] > 0, 'early_return'] = 1
    df.loc[df['early_return_amount'] > 0, 'early_return'] = 1
    df.loc[df['early_return_amount'] == 0, 'early_return'] = 0

    if 'house_loan_status' in df.columns:
        upper_house_loan_status = 2
        lower_house_loan_status= 0
        #  df['house_loan_status'] = df['house_loan_status'].apply(lambda x: x if x < upper_house_loan_status*10 else np.nan)
        df['house_loan_status'] = df['house_loan_status'].apply(lambda x: x if x <= upper_house_loan_status else np.nan)
        #  df.loc[:,'house_loan_status'] = df.loc[:,'house_loan_status'].apply(lambda x: x if x > 0 else np.nan)
        df['house_loan_status'] = df['house_loan_status'].apply(lambda x: x if x > lower_house_loan_status else np.nan)


    if 'is_default' in df.columns:
        #  put the label to the last column
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)
    
    return df
    
def feature_extract(df1): 
    df = df1.copy()

    if ('monthly_payment' in df.columns) & ('year_of_loan' in df.columns) & ('debt_loan_ratio' in df.columns):
        df['income'] = df.eval('monthly_payment * year_of_loan / debt_loan_ratio')

    if 'offsprings' in df.columns:
        df['havent_springs'] = df['offsprings'].apply(lambda x: 1 if x > 0 else x)

    if 'class' in df.columns:
        df['class'] = df['class'].apply(lambda x: (x - 65)*6)
        if 'sub_class' in df.columns:
            df['class'] = np.where(df['sub_class'].notnull(), df['class'] + df['sub_class'], df['class'])
            df = df.drop(['sub_class'], axis = 1)

    log_list = ['total_loan', 'interest', 'monthly_payment', 'debt_loan_ratio', 'del_in_18month', 'known_outstanding_loan', 'recircle_b', 'recircle_u', 'f0', 'f2', 'f3', 'f4', 'f5', 'early_return_amount', 'early_return_amount_3mon', 'income']

    for col in log_list:
        if col in df.columns:
            df[col] = np.log1p(df[col])


    #  if 'marriage' in df.columns:
    #      df['havent_marrage'] = df['marriage'].apply(lambda x: 1 if x > 0 else x)

    #  put the label to the last column
    if 'is_default' in df.columns:
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)

    return df

def binning(df1, binning_path = None):
    df = df1.copy()

    number_of_bins_of_class = 5
    number_of_bins_of_issue_year = 8
    number_of_bins_of_house_exist = 4
    number_of_bins_of_earlies_credit_mon            = 7

    #  issue_year_labels = range(number_of_bins_of_issue_year)
    #  house_exist_labels = range(number_of_bins_of_house_exist)
    #  earlies_credit_mon_labels = range(number_of_bins_of_earlies_credit_mon)


    bin_edges = {} 


    if binning_path is None:
        if 'class' in df.columns:
            df['class_bin'], bin_edges['class'] = pd.qcut(df['class'], q=number_of_bins_of_class,labels= False, retbins=True, duplicates='drop')
            df = df.drop(['class'], axis = 1)
        if 'issue_year' in df.columns:
            df['issue_year_bin'], bin_edges['issue_year'] = pd.qcut(df['issue_year'], q=number_of_bins_of_issue_year,labels= False, retbins=True, duplicates='drop')
            df = df.drop(['issue_year'], axis = 1)
        if 'house_exist' in df.columns:
            df['house_exist_bin'], bin_edges['house_exist'] = pd.qcut(df['house_exist'], q=number_of_bins_of_house_exist, labels= False, retbins=True, duplicates= 'drop')
            df = df.drop(['house_exist'], axis = 1)
        if 'earlies_credit_mon' in df.columns:
            df['earlies_credit_mon_bin'], bin_edges['earlies_credit_mon'] = pd.qcut(df['earlies_credit_mon'], q=number_of_bins_of_earlies_credit_mon,labels= False, retbins=True, duplicates= 'drop')
            df = df.drop(['earlies_credit_mon'], axis = 1)

    else:
        with open(binning_path, 'rb') as f:
            bin_edges = pickle.load(f) 


        if 'class' in df.columns:
            df['class_bin'] = pd.cut(df['class'], bins=bin_edges['class'], labels= False)
            df = df.drop(['class'], axis = 1)
        if 'issue_year' in df.columns:
            df['issue_year_bin'] = pd.cut(df['issue_year'], bins=bin_edges['issue_year'], labels= False)
            df = df.drop(['issue_year'], axis = 1)
        if 'house_exist' in df.columns:
            df['house_exist_bin'] = pd.cut(df['house_exist'], bins=bin_edges['house_exist'], labels= False)
            df = df.drop(['house_exist'], axis = 1)
        if 'earlies_credit_mon' in df.columns:
            df['earlies_credit_mon_bin'] = pd.cut(df['earlies_credit_mon'], bins=bin_edges['earlies_credit_mon'], labels= False)
            df = df.drop(['earlies_credit_mon'], axis = 1)

    #  put the label to the last column
    if 'is_default' in df.columns:
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)

    return df, bin_edges

def encoding(df1):
    df = df1.copy()

    encoding_list = ['work_type', 'employer_type', 'industry', 'censor_status', 'use', 'marriage', 'issue_dayofweek', 'issue_year_bin', 'house_exist_bin', 'earlies_credit_mon_bin']




    for col in encoding_list:
        if col in df.columns:
            df = one_hot_encoding(df, col)

    #  put the label to the last column
    if 'is_default' in df.columns:
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)

    return df

def embedding(df1, embedding_path):
    df = df1.copy()

    max_number_of_cats_of_post_code = 8
    max_number_of_cats_of_region = 8
    max_number_of_cats_of_title = 8

    with open(embedding_path, 'rb') as f:
        #  issue_year_proportions = pickle.load(f)
        #  issue_dayofweek_proportions = pickle.load(f)
        #  use_proportions = pickle.load(f)
        post_code_proportions = pickle.load(f)
        region_proportions = pickle.load(f)
        title_proportions = pickle.load(f)
        #  earlies_credit_mon_proportions = pickle.load(f)

        #  issue_year_dict = pickle.load(f)
        #  issue_dayofweek_dict = pickle.load(f)
        #  use_dict = pickle.load(f)
        post_code_dict = pickle.load(f)
        region_dict = pickle.load(f)
        title_dict = pickle.load(f)
        #  earlies_credit_mon_dict = pickle.load(f)

    embedding_list = ['post_code', 'region', 'title']
    if 'post_code' in df.columns:
        df = extract_category(df, 'post_code', post_code_proportions, post_code_dict, max_number_of_cats_of_post_code)
    if 'region' in df.columns:
        df = extract_category(df, 'region', region_proportions, region_dict, max_number_of_cats_of_region)
    if 'title' in df.columns:
        df = extract_category(df, 'title', title_proportions, title_dict, max_number_of_cats_of_title)


    #  put the label to the last column
    if 'is_default' in df.columns:
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)

    return df

def embed_category(df1, col, col_proportions, col_dict, n):
    #  print(col_proportions)
    #  print(col_dict)
    df = df1.copy()
    p = 1.0 / n

    #  cols = []
    #  current_categories = []
    #  current_col_proportion = 0.0
    #  remaining_proportion = 1.0
    cats = []
    cats_dict = {}

    print(col)

    i = 0
    for category in col_proportions:
        cats.append(category)
        if col_proportions[category] > p:
            col_dict[category] = i
            i = i + 1
            print(category)
        else:
            col_dict[category] = i

    print(col_dict)
    #  print(cats)

        #  proportion = col_proportions[category]
        #  current_categories.append(category)
        #  current_col_proportion += proportion
        #  if (current_col_proportion >= p):
        #      cols.append(list(current_categories))
        #      remaining_proportion -= current_col_proportion
        #      current_categories = []
        #      current_col_proportion = 0.0
                
    #  cols.append(list(current_categories))
            
    #  new_df = pd.DataFrame()
    #  for i, categories in enumerate(cols):
    df[col] = df[col].apply(lambda x: col_dict[x] if x in cats else -1)
        #  new_df[col + '_' + str(i+1)] = df[col].apply(lambda x: col_dict[x] if x in categories else 0)
        
    #  df = pd.concat([df, new_df], axis= 1)
    #  df = df.drop([col], axis = 1)

    return df

def cat_embedding(df1, embedding_path):
    df = df1.copy()

    max_number_of_cats_of_post_code = 20
    max_number_of_cats_of_region = 20
    max_number_of_cats_of_title = 20
    with open(embedding_path, 'rb') as f:
        #  issue_year_proportions = pickle.load(f)
        #  issue_dayofweek_proportions = pickle.load(f)
        #  use_proportions = pickle.load(f)
        post_code_proportions = pickle.load(f)
        region_proportions = pickle.load(f)
        title_proportions = pickle.load(f)
        #  earlies_credit_mon_proportions = pickle.load(f)

        #  issue_year_dict = pickle.load(f)
        #  issue_dayofweek_dict = pickle.load(f)
        #  use_dict = pickle.load(f)
        post_code_dict = pickle.load(f)
        region_dict = pickle.load(f)
        title_dict = pickle.load(f)
        #  earlies_credit_mon_dict = pickle.load(f)

    embedding_list = ['post_code', 'region', 'title']
    if 'post_code' in df.columns:
        df = embed_category(df, 'post_code', post_code_proportions, post_code_dict, max_number_of_cats_of_post_code)
    if 'region' in df.columns:
        df = embed_category(df, 'region', region_proportions, region_dict, max_number_of_cats_of_region)
    if 'title' in df.columns:
        df = embed_category(df, 'title', title_proportions, title_dict, max_number_of_cats_of_title)


    #  put the label to the last column
    if 'is_default' in df.columns:
        cols = list(df.columns)
        cols.remove('is_default')
        cols.append('is_default')
        df = df.reindex(columns=cols)

    return df







def aligning(df1, aligning_path):
    df = df1.copy()

    with open(aligning_path, 'r') as f:
        cols = f.read().strip().split(',')
    
    df = df.reindex(columns=cols)
    return df


def normalization (df1, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    df = df1.copy()

    if parameters is None:
  
        # MixMax normalization
        min_val = {}
        max_val = {}

        # For each dimension
        for col in df.columns:
            min_val[col] = np.nanmin(df[col])
            df[col] = df[col].apply(lambda x: x - min_val[col])
            max_val[col] = np.nanmax(df[col])
            if max_val[col] == 0:
                df[col] = df[col].apply(lambda x: 0)
            else: 
                df[col] = df[col].apply(lambda x: x / max_val[col])
          
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                               'max_val': max_val}

    else:
        
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        #  print(max_val)

        # For each dimension
        for col in df.columns:
            df[col] = df[col].apply(lambda x: min_val[col] if x < min_val[col] else x)
            df[col] = df[col].apply(lambda x: x - min_val[col])
            df[col] = df[col].apply(lambda x: min_val[col] if x > max_val[col] else x)
            if max_val[col] == 0:
                df[col] = df[col].apply(lambda x: 0)
            else: 
                df[col] = df[col].apply(lambda x: x / max_val[col])
        norm_parameters = parameters    

    df = pd.DataFrame(df)
  
    return df, norm_parameters


def standardization(df1, parameters=None):
    '''standardize data 

    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    df = df1.copy()

    if parameters is None:
  
        # MixMax normalization
        mean_val = {}
        std_val = {}

        # For each dimension
        for col in df.columns:
            mean_val[col] = df[col].mean()
            std_val[col] = df[col].std()
            df[col] = df[col].apply(lambda x: x - mean_val[col])
            if std_val[col] == 0:
                df[col] = df[col].apply(lambda x: 0)
            else: 
                df[col] = df[col].apply(lambda x: x / std_val[col])
          
        # Return norm_parameters for renormalization
        std_parameters = {'mean_val': mean_val,
                               'std_val': std_val}

    else:
        
        mean_val = parameters['mean_val']
        std_val = parameters['std_val']
        #  print(max_val)

        # For each dimension
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x - mean_val[col])
            if std_val[col] == 0:
                df[col] = df[col].apply(lambda x: 0)
            else: 
                df[col] = df[col].apply(lambda x: x / std_val[col])
        std_parameters = parameters    

    df = pd.DataFrame(df)
  
    return df, std_parameters
    

def fill_nan_with_neg(df1):
    df = df1.copy()
    for col in df.columns:
        df[col].fillna(-1, inplace=True)
    return df


#  from sklearn.cluster import OPTICS
#  def cluster(df1, col_list, model = None):
#
#      data = df1[col_list]
#
#      if model is None:
#          model = OPTICS(min_samples=2)
#          model.fit(data)
#          result = model.labels_
#
#      else:
#          result = model.predict(data)
#
#      return result, model
#
#  def clusterlize(df1, model_path = None):
#      df = df1.copy()
#
#      cols1= ['post_code', 'region']
#      #  cols2 = ['work_type', 'industry', 'employer_type']
#
#      limit1 = 30
#
#      if model_path is None:
#          result1, model1 = cluster(df, cols1)
#          #  result1, model1 = cluster(df, cols1)
#
#      else:
#          with open(model_path, 'rb') as f:
#              #  issue_year_proportions = pickle.load(f)
#              #  issue_dayofweek_proportions = pickle.load(f)
#              #  use_proportions = pickle.load(f)
#              model1 = pickle.load(f)
#
#          result1, _ = cluster(df, cols1, model)
#
#      return df
#
#
#      with open(embedding_path, 'wb') as f:
#          #  pickle.dump(issue_year_proportions, f)
#          #  pickle.dump(issue_dayofweek_proportions, f)
#          #  pickle.dump(use_proportions, f)
#          pickle.dump(post_code_proportions, f)
#
#
#
#
#
#










