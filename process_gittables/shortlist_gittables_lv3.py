import os
from shortlist_utils import *

DATA_PATH = '../dataset_gittables'
SAVE_PATH = '../eda_gittables'

# all paths of dataset directories
paths = os.listdir(SAVE_PATH)

for path in paths:
    
    print(path)
    
    # each dataset dir path 
    eda_path = os.path.join(SAVE_PATH, path)
    # data_path = os.path.join(DATA_PATH, path)
    
    # lv 3.1
    stats_df_lv2 = pd.read_csv(os.path.join(eda_path, 'stats_lv2.csv'))
    stats_df_lv3 = group_ds(stats_df_lv2)
    stats_df_lv3.to_csv(os.path.join(eda_path,  'stats_lv3.csv'))
    
    # lv 3.2 - group datasets still being single
    stats_df_lv3_2 = group_single_df(stats_df_lv3, DATA_PATH, path)
    stats_df_lv3.to_csv(os.path.join(eda_path,  'stats_lv3_2.csv'))
