import os
import shutil
import pandas as pd
from preprocess.preprocessing import preprocess_clean_gittables

############# CONFIG #############

SOURCE_DIR = 'data/gittables_v2' # data source
TARGET_DIR = 'data/gittables_v3' # target directory to save processed data

PREPROCESS_CONFIG = {
    'clean_data': True,
    'clean_exclude_columns': [], # exclude chosen columns to apply cleaning, if empty, all columns are considered
    'min_freq_threshold': 0.08, # if max frequency of a values of a column >= this param, all values of the column will be kept
    'percentage_to_remove': 0.9, # if the categories is too sparse (a lot of categories but very small number of data per categoru) | if the number of categories in the column >= this param * number of data rows, it will be removed
    'percentage_to_remove_row': 0.5, # if nan values are too many (>= this param * n_rows), remove
    'generate_metadata': True,
}

############# END CONFIG #############

# data_dirs = []
# paths = os.listdir(EDA_DIR)

# for path in paths:

#     # each dataset dir path 
#     eda_path = os.path.join(EDA_DIR, path)
#     stats_df = pd.read_csv(os.path.join(eda_path, 'stats_lv2.csv'))
#     dt_dirs = stats_df['dataset'].apply(lambda k : os.path.join(SOURCE_DIR, path, k)).to_list()
#     data_dirs += dt_dirs
    
# print(f'Total data dirs: {len(data_dirs)}')

# LOAD DATA
# get all sub dir csv file
sub_dirs = os.listdir(SOURCE_DIR)
# sub_dirs = [k for k in sub_dirs if '.DS_Store' not in k]

# get all csv file and meta data in sub dir
data_dirs = [[os.path.join(SOURCE_DIR, _dir, k) \
    for k in os.listdir(os.path.join(SOURCE_DIR, _dir)) \
        if '.csv' in k] for _dir in sub_dirs \
            if os.path.isdir(os.path.join(SOURCE_DIR, _dir))]

data_dirs = [item for sublist in data_dirs for item in sublist]
print('All csv file in source domain: ', data_dirs)

# REMOVE DUPLICATE
# preprocess_duplicate(data_dirs, target_domain_dir, verbose=0)


# print('=====================AUTO CLEAN & GENERATE METADATA========================')
preprocess_clean_gittables(data_dirs, TARGET_DIR, PREPROCESS_CONFIG, min_cols_to_keep=2, min_rows_to_keep=10, verbose=0)


# print('=====================TRANSFORM========================')
# preprocess_transform(target_domain_dir, transform_cfg, verbose=0)


# # remove data folder if no clean_info
# for _dir in data_dirs:
#     data_dir = os.path.dirname(_dir)
#     clean_info_path = os.path.join(data_dir, 'clean_info.json')
    
#     if not os.path.exists(clean_info_path):
#         os.system(f"rm -rf '{data_dir}'")
        
#         print(f'\t - Removed {data_dir}')
