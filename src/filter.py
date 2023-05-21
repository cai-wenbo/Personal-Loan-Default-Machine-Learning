import sys

import pandas as pd
import numpy as np
from pandas._libs.tslibs.period import from_ordinals
from pandas.core.indexes.base import default_pprint
pd.options.display.max_rows = None

from src.neuro import classifier

def filte(df1, filter_path, limit):    
    df  = df1.copy() 
    result = classifier(df,  'model/filter/public_filter.pt')

    result = result.rename(columns={'is_default': 'judge'})
    df = pd.concat([df, result], axis = 1)
    df['diff'] = abs(df['judge'] - df['is_default'])
    counts_1 = df['is_default'].value_counts()[1]
    portion_1 = counts_1 / len(df)
    print(portion_1)
    selected_df = df[df['diff'] < limit].copy()

    print(len(selected_df) * 1.0/len(df))
    selected_df = selected_df.drop(['judge', 'diff'], axis = 1)

    counts_1 = selected_df['is_default'].value_counts()[1]
    portion_1 = counts_1 / len(selected_df)
    print(portion_1)
    return selected_df

