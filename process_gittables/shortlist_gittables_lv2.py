import os
from shortlist_utils import *

DATA_PATH = 'dataset_gittables'
SAVE_PATH = 'eda_gittables'

print('#################### Shortlist gittables lv2 ####################')
# all paths of dataset directories
paths = os.listdir(DATA_PATH)
print('current data dir: ', DATA_PATH)
print('all paths: ', paths)
for path in paths:
    print(path)
    
    # each dataset dir path 
    eda_path = os.path.join(SAVE_PATH, path)
    data_path = os.path.join(DATA_PATH, path)
    
    stats_df_lv1 = pd.read_csv(os.path.join(eda_path, 'stats.csv'))
    slist_df_lv2 = shortlist_gittables_lv2(stats_df_lv1, data_path)
    
    n_data = len(stats_df_lv1)
    n_processed_data = len(slist_df_lv2)
    
    # # shortlist
    # try:
    #     slist_df = shortlist_gittables_lv1(data_path)
    # except:
    #     continue
    
    print(f'\t Shortlisted {n_processed_data} / {n_data} | {round((n_processed_data / n_data) * 100, 2)}%')
    
    slist_df_lv2.to_csv(os.path.join(eda_path,  'stats_lv2.csv'))
