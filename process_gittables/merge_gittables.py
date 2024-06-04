import os
from shortlist_utils import merge_df
import pandas as pd
import gc

DATA_PATH = '../dataset_gittables'
EDA_PATH = '../eda_gittables'
SAVE_PATH = 'data/gittables_v2'

paths = os.listdir(EDA_PATH)

for path in paths:
    print(f'SET {path}--------------------------')
    
    eda_path = os.path.join(EDA_PATH, path)
    stats_df_lv3 = pd.read_csv(os.path.join(eda_path, 'stats_lv3_2.csv'))
    
    merge_df(stats_df_lv3, path, DATA_PATH, SAVE_PATH)
    
    gc.collect()