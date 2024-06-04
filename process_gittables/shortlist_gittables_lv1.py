import os
from shortlist_utils import *

DATA_PATH = '../dataset_gittables'
SAVE_PATH = '../eda_gittables'

# all paths of dataset directories
paths = os.listdir(DATA_PATH)


for path in paths:
    
    if not os.path.isdir(path):
        continue
    
    print(path)
    
    stats_save_path = os.path.join(SAVE_PATH, path)
    
    if os.path.exists(stats_save_path):
        continue
    
    
    # each dataset dir path 
    data_path = os.path.join(DATA_PATH, path)
    n_data = len(os.listdir(data_path))

    # shortlist
    try:
        slist_df = shortlist_gittables_lv1(data_path)
    except:
        continue
    
    print(f'\t Shortlisted {len(slist_df)} / {n_data} | {round((len(slist_df) / n_data) * 100, 2)}%')
    
    os.mkdir(stats_save_path)
    slist_df.to_csv(os.path.join(stats_save_path,  'stats.csv'))

