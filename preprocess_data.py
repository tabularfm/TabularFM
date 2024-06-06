import os
import shutil
from tabularfm.preprocess.preprocessing import preprocess_clean

############# CONFIG #############

SOURCE_DIR = 'data/processed_dataset' # data source
TARGET_DIR = 'data/processed_dataset' # target directory to save processed data

PREPROCESS_CONFIG = {
    'clean_data': True,
    'clean_exclude_columns': [], # exclude chosen columns to apply cleaning, if empty, all columns are considered
    'min_freq_threshold': 0.03, # if max frequency of a values of a column >= this param, all values of the column will be kept
    'percentage_to_remove': 0.9, # if the number of categories in the column >= this param * number of data rows, it will be removed
    'percentage_to_remove_row': 0.5,
    'generate_metadata': True,
}

############# END CONFIG #############

# LOAD DATA
# get all sub dir csv file
sub_dirs = os.listdir(SOURCE_DIR)
sub_dirs = [k for k in sub_dirs if '.DS_Store' not in k]

# get all csv file and meta data in sub dir
data_dirs = [[os.path.join(SOURCE_DIR, _dir, k) \
    for k in os.listdir(os.path.join(SOURCE_DIR, _dir)) \
        if '.csv' in k] for _dir in sub_dirs \
            if os.path.isdir(os.path.join(SOURCE_DIR, _dir))]

data_dirs = [item for sublist in data_dirs for item in sublist]
print('All csv file in source domain: ', data_dirs)

# REMOVE DUPLICATE
# preprocess_duplicate(data_dirs, target_domain_dir, verbose=0)


print('=====================AUTO CLEAN & GENERATE METADATA========================')
preprocess_clean(data_dirs, TARGET_DIR, PREPROCESS_CONFIG, min_cols_to_keep=2, min_rows_to_keep=10, verbose=0)


# print('=====================TRANSFORM========================')
# preprocess_transform(target_domain_dir, transform_cfg, verbose=0)