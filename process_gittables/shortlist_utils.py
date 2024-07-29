import pandas as pd
import csv
import os
import numpy as np
import re
from pathlib import Path
import pickle
import re

# SHORTLIST LV1
# find_prefixes
def find_prefixes(stats_df):
    words = stats_df['dataset'].apply(lambda k : k.split('.')[0]).to_list()
    result = sorted([ (sum([ w.startswith(prefix) for w in words ]) , prefix )  for prefix in words])[::-1]
    
    return result

def get_df(path):
    try:
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", index_col=0)
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1")
        df = pd.read_csv(path, on_bad_lines='skip', encoding = 'utf-8')
    except:
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", lineterminator='\n', quoting=csv.QUOTE_NONE, index_col=0)
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", quoting=csv.QUOTE_NONE, index_col=0)
        df = pd.read_csv(path, on_bad_lines='skip', encoding = 'utf-8', lineterminator='\n', quoting=csv.QUOTE_NONE)
        
    return df

# get stats of datasets list
def get_stats(data_path, verbose):
    paths = os.listdir(data_path)
    rs = []
    for path in paths:
        
        if 'csv' != path.split('.')[-1]:
            continue
        
        path = os.path.join(data_path, path)
        if verbose:
            print(path)
            
        df = get_df(path)
        
        # df = pd.read_parquet(path, engine='pyarrow')
        n_rows = len(df)
        n_cols = len(df.columns)
        
        rs.append({
            'dataset': os.path.basename(path),
            'rows': n_rows,
            'columns': n_cols,
        })
        
    stats_df = pd.DataFrame(rs)
    
    return stats_df

def get_stats_by_prefix(stats_df, prefix):
    rs_df = stats_df[stats_df['dataset'].apply(lambda k: True if prefix in k else False)]
    
    return rs_df

def shortlist_gittables_lv1(data_path, verbose=False):
    stats_df = get_stats(data_path, verbose)
    
    # get dataset with cols >= q1
    q1_col = stats_df['columns'].quantile(0.25)
    shortlist_stats_df = stats_df[stats_df['columns'] >= q1_col]
    
    # # find datasets with the same prefix to recalculate the #rows
    # prefixes = find_prefixes(shortlist_stats_df)
    # prefixes = pd.Series([k[1] for k in prefixes]).value_counts()
    # prefixes = prefixes[prefixes > 1].keys().to_list()
    
    return shortlist_stats_df

# SHORTLIST LV2

def _is_free_text_df(df, threshold=0.5):
    cat_df = df[list(set(df.columns) - set(df._get_numeric_data().columns))]

    if len(cat_df) == 0:
        return False
    
    avg_free_text = np.mean([cat_df[col].nunique() / len(cat_df) for col in cat_df.columns])
    avg_free_text = avg_free_text * (len(cat_df.columns) / len(df.columns))  # multiply with ratio of cat columns / all columns
    
    if avg_free_text >=threshold:
        return True
    
    return False

def _is_large_na_df(df, threshold=0.6):
    na_ratio = np.mean(df.isnull().sum() / len(df))
    
    if na_ratio >= threshold:
        return True
    
    return False

def _convert_num_string_to_numeric(df):
    '''
    Converts common types of numeric strings such as money values, percentages, etc. to numeric values.
    Suggested workflow: since this function will return NaN if a value is not valid,
    we recommend dropping/imputing empty values AFTER this is done.
    '''
    def _get_num_from_string(s):
        # we assume that the numerical values will only ever contain one '.'
        # if not, we will return nothing
        # O(n) string processing
        num = ''
        # only these will directly impact the downstream parsing
        allowed_characters = {
            '.': 0,
            '-': 0,
            'e': 0,
        }
        for i in range(len(s)):
            if s[i] == '.':
                allowed_characters['.'] += 1
                # handle multiple invalid misinputs
                if allowed_characters['.'] > 1:
                    return None
                num += s[i]
            elif s[i] == '-' and i == 0:
                allowed_characters['-'] += 1
                # handle cases like phone numbers separated by '-'
                if allowed_characters['-'] > 1:
                    return None
                num += s[i]
            elif s[i] == 'E' or s[i] == 'e':
                allowed_characters['e'] += 1
                # handle cases like 'eeeeeeeeee'
                if allowed_characters['e'] > 1:
                    return None
                num += s[i]
            elif s[i].isdigit():
                num += s[i]
        
        if not num: return None
        try: 
            num = float(num)
            return num
        except Exception as e:
            # add custom logging functions here
            print('Error converting string to float:', e)
            return None
    
    def _check_if_mostly_numeric(column):
        # get 10 possible samples
        samples = df[column].sample(10)
        success = 0
        pattern = r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$'

        for sample in samples: 
            # remove all whitespaces, commas, and parentheses
            sample = sample.strip().replace(',', '').replace(' ', '').replace('(', '').replace(')', '')
            # matches negatives, decimals, and scientific notation
            if re.match(pattern, sample):
                success += 1
            elif sample.endswith('%'):
                sample = sample[:-1]
                if re.match(pattern, sample):
                    success += 1
            elif re.match(r'^[\$\£\€]', sample):
                sample = sample[1:]
                if re.match(pattern, sample):
                    success += 1

        # needs 60% success rate to continue
        return success >= 6
    
    for column in df.columns: 
        if df[column].dtype == 'object' and _check_if_mostly_numeric(column):
            df[column] = df[column].apply(lambda x: pd.to_numeric(_get_num_from_string(x), errors='coerce'))
    
    return df


def shortlist_gittables_lv2(shortlist_stats_df, data_path):
    ds_names = shortlist_stats_df['dataset'].to_list()
    
    # def _is_nonsense_colname(df):
    #     df.columns.to_list()
    
    valid_mask = []
    for ds_name in ds_names:
        
        df = get_df(os.path.join(data_path, ds_name))
        
        if _is_free_text_df(df):
            valid_mask.append(False)
            continue
        
        if _is_large_na_df(df):
            valid_mask.append(False)
            continue
        
        valid_mask.append(True)
        
    return shortlist_stats_df[valid_mask]

# SHORTLIST LV3

def has_no_letters(text):
    has_letter = bool(re.search('[a-zA-Z]', Path(text).stem))
    
    if has_letter:
        return False
    
    return True


def set_group(stats_df, filter_list, groupname):
    gr_df = stats_df[stats_df['dataset'].isin(filter_list)].reset_index(drop=True)
    gr_df['group'] = groupname
    
    return gr_df

def group_ds(stats_df):
    groupcol_ds = stats_df.groupby(['columns'])['dataset'].apply(list).to_dict()
    
    stats_df_lv3 = []
    for _, v in groupcol_ds.items():
        
        # handle all number case
        mask = np.array([True if has_no_letters(k) else False for k in v])
        
        if mask.sum() > 0:
            num_ds_names = np.array(v)[mask]
            gr_df = set_group(stats_df, num_ds_names, 'number')
            stats_df_lv3.append(gr_df)
            
            v_rest = np.array(v)[~mask]

        else:
            v_rest = v
        
        # handle other case
        prefixes = np.unique([k.split('_')[0] for k in v_rest])
        
        for prefix in prefixes:
            group_ds_names = [k for k in v_rest if prefix in k]
            
            gr_df = set_group(stats_df, group_ds_names, prefix)
            stats_df_lv3.append(gr_df)
            
    # concat and remove duplicates
    stats_df_lv3 = pd.concat(stats_df_lv3)
    stats_df_lv3 = stats_df_lv3.drop_duplicates(subset='dataset')
    
    return stats_df_lv3

# GROUP SINGLE DF
def update_group(stats_df, ds_names, group_name):
    update_index = stats_df.index[stats_df['dataset'].isin(ds_names)]
    stats_df.loc[update_index, 'group'] = group_name
    
def match_by_same_columns(lst):
    unique_lists = []
    all_indices = {}
    for i, sublist in enumerate(lst):
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in unique_lists:
            unique_lists.append(sublist_tuple)
        if sublist_tuple not in all_indices:
            all_indices[sublist_tuple] = [i]
        else:
            all_indices[sublist_tuple].append(i)
    return [list(sublist) for sublist in unique_lists], all_indices

def group_single_df(stats_df, data_path, path):
    
    # find ds groups are single
    single_ds = list(stats_df['group'].value_counts()[list(stats_df['group'].value_counts() == 1)].keys())
    single_df = stats_df[stats_df['dataset'].isin(single_ds)]
    group_single_ds = single_df.groupby('columns')['dataset'].apply(list).to_dict()
    
    # merge if their columns are identical
    for single_idx, (_,v) in enumerate(group_single_ds.items()):
        columns = [get_df(os.path.join(data_path, path, ds)).columns for ds in v]
        
        # find datasets where columns are identical
        _, match_info = match_by_same_columns(columns)
        
        # group them together
        group_ds_names = [np.array(v)[indices] for _, indices in match_info.items()]
        
        # only get which group has len > 1
        group_ds_names = [k for k in group_ds_names if len(k) > 1]
        
        # update their group names
        for gr_name in group_ds_names:
            update_group(stats_df, gr_name, f'single_{single_idx}')
            
    return stats_df


# MERGE DF

# def merge_large_df(df_paths):
#     mtot=0
#     with open('df_all.bin','wb') as f:
#         for df_path in df_paths:
#             df = get_df(df_path)
#             m,n =df.shape
#             mtot += m
#             f.write(df.values.tobytes())
#             typ=df.values.dtype                
#     #del dfs
#     with open('df_all.bin','rb') as f:
#         buffer=f.read()
#         data=np.frombuffer(buffer,dtype=typ).reshape(mtot,n)
#         df_all=pd.DataFrame(data=data,columns=list(range(n))) 
#     os.remove('df_all.bin')
    
#     return df_all

# def merge_large_df(df_paths):
#     c=[]
#     with open('df_all.pkl','ab') as f:
#         for df_path in df_paths:
#             df = get_df(df_path)
#             pickle.dump(df,f)
#             c.append(len(df))    
#     #del dfs
#     with open('df_all.pkl','rb') as f:
#         df_all=pickle.load(f)
#         offset=len(df_all)
#         # df_all=df_all.append(pd.DataFrame(np.empty(sum(c[1:])*4).reshape(-1,4)))
#         df_all = pd.concat([df_all, pd.DataFrame(np.empty(sum(c[1:])*4).reshape(-1,4))])
        
#         for size in c[1:]:
#             df=pickle.load(f)
#             df_all.iloc[offset:offset+size]=df.values 
#             offset+=size
#     os.remove('df_all.pkl')
#     return df_all

# def merge_large_df(df_paths):
#     store=pd.HDFStore('df_all.h5')
#     for df_path in df_paths:
#         df = get_df(df_path)
#         store.append('df',df,data_columns=df.columns)
#     #del dfs
#     df=store.select('df')
#     store.close()
#     os.remove('df_all.h5')
#     return df

def merge_large_df(df_paths):
    md,hd='w',True
    for df_path in df_paths:
        df = get_df(df_path)
        df.to_csv('df_all.csv',mode=md,header=hd,index=None)
        md,hd='a',False
    #del dfs
    df_all = get_df('df_all.csv')
    os.remove('df_all.csv') 
    return df_all

def iterate_save_large_df(df, iter, csv_path):
    md,hd='a',False
    
    if iter == 0:
        md,hd='w',True
        
    df.to_csv(csv_path,mode=md,header=hd,index=None)
    
def merge_df(stats_df, path, data_path, save_path):
    import gc
    
    groups = stats_df.groupby('group')['dataset'].apply(list).to_dict()

    for gr_name, gr_paths in groups.items():
        gr_name = Path(gr_name).stem
        print(f'Group {gr_name}')
        # try:
        if 'single' in gr_name or 'number' in gr_name: # large file
            merged_df = merge_large_df([os.path.join(data_path, path, ds_name) for ds_name in gr_paths])
        else:
            merged_df = pd.concat([get_df(os.path.join(data_path, path, ds_name)) for ds_name in gr_paths]) 
                
        # except:
        #     print('\t - Failed')
        #     continue
        
        gr_name_wpath = path + '_' + gr_name
        save_ds_dir = os.path.join(save_path, gr_name_wpath)    
        if not os.path.exists(save_ds_dir):
            os.mkdir(save_ds_dir)
        
        save_csv_path = os.path.join(save_ds_dir, gr_name_wpath + '.csv')
        merged_df.to_csv(save_csv_path)
        
        gc.collect()