import os
import pickle
import json
import numpy as np
import pandas as pd

from preprocess.cleaning import TabCleaning
from preprocess.meta import SingleTabMetadata


def preprocess_duplicate(data_dirs, target_domain_dir, verbose=0):
    headers = np.array([k.split('/')[-2] for k in data_dirs])
    headers = np.unique(headers)
    
    if verbose: print('Number of unique datasets: ', len(headers))
    
    pass

def get_id_column_name(clean_info):
    for col_name, value in clean_info.items():
        if value['desc'] == 'ID':
            return col_name
        
    return None
        
def preprocess_clean(src_data_dirs, dst_data_dirs, preprocessing_cfg, min_cols_to_keep=2, min_rows_to_keep=100, verbose=0):
    """Apply TabCleaning to all data, save clean_df in the same directory with prefix 'clean_'

    Args:
        data_dirs ([type]): [description]
        preprocessing_cfg ([type]): [description]
    """
    
    for _dir in src_data_dirs:
        print(20* '-')
        print('_dir: ', _dir)
        
        # df = pd.read_csv(_dir)
            
        # GENERATE METADATA
        if preprocessing_cfg['generate_metadata']:
            print('- Generate metata')
            meta = SingleTabMetadata(df)
            metadata = meta.get_metadata(verbose=verbose)
            
        # CLEAN DATA
        if preprocessing_cfg['clean_data']:
            print('- Clean data')
            df, clean_info = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, 
                                                                                    min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                    pct_to_remove=preprocessing_cfg['percentage_to_remove'],
                                                                                    pct_to_remove_row=preprocessing_cfg['percentage_to_remove_row'],
                                                                                    transform_stringified_numerical_columns=preprocessing_cfg['transform_stringified_numerical_columns'],
                                                                                    return_info=True,
                                                                                    verbose=verbose)
            
            print('DEBUG clean info: ', clean_info)
            
            # skip this df if only 1 column
            if len(df.columns) <= min_cols_to_keep:
                print('\t Skip because small number of columns: ', len(df.columns))
                continue
            
            # skip this df if number of rows <= 100
            if len(df) <= min_rows_to_keep:
                print('\t Skip because small number of row: ', len(df))
                continue
            
        # DENOTE ID COLUMN IN METADATA
        id_col_name = get_id_column_name(clean_info)
        
        if id_col_name is None:
            # id_col_name = list(clean_info.keys())[0]
            meta.metadata['primary_key'] = id_col_name
        
        else:
            meta.metadata['primary_key'] = id_col_name
            meta.metadata['columns'][id_col_name]['sdtype'] = "id"
            
        # SAVE
        # save dir setup
        save_path = os.path.join(dst_data_dirs, os.path.dirname(_dir).split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # save metadata
        if preprocessing_cfg['generate_metadata']:
            prefix_metadta = ''
            meta.to_json(os.path.join(save_path, 'metadata.json'))
        
        # save cleaned df and clean info 
        if preprocessing_cfg['clean_data']:
            prefix_clean = ''
            df.to_csv(os.path.join(save_path, prefix_clean + os.path.dirname(_dir).split('/')[-1] + '.csv'), index=False)
            json.dump(clean_info, open(os.path.join(save_path, 'clean_info.json'), 'w'))
            
    return None

def chunk_df(df_path, chunksize=1e4, load_df_func=None):
    load_df = load_df_func if load_df_func is not None else pd.read_csv
    df_size = os.path.getsize(df_path) / 1e6
    if df_size >= 200: # large file, split
        for chunk in load_df(df_path, chunksize=chunksize):
            yield chunk
    else:
        yield load_df(df_path)
        
def aggregate_clean_info(clean_info_chunks):
    agg_clean_info = clean_info_chunks[0].copy()
    
    # update when any clean_info reject to remove
    for col_name, _ in agg_clean_info.items():
        for clean_info in clean_info_chunks[1:]:
            if clean_info[col_name]["keep"] == True:
                agg_clean_info[col_name]["keep"] = True
                agg_clean_info[col_name]["desc"] = "NA"
                break
    
    return agg_clean_info


def preprocess_clean_gittables(src_data_dirs, dst_data_dirs, preprocessing_cfg, min_cols_to_keep=2, min_rows_to_keep=100, verbose=0):
    """Apply TabCleaning to all data, save clean_df in the same directory with prefix 'clean_'

    Args:
        data_dirs ([type]): [description]
        preprocessing_cfg ([type]): [description]
    """
    from shortlist_utils import iterate_save_large_df
    error_datasets = []
    
    for _dir in src_data_dirs:
        print(20* '-')
        print('_dir: ', _dir)
        
        # try:
        # df = pd.read_csv(_dir)
        from shortlist_utils import get_df
        
        clean_info_chunks = []
        n_keep_rows = 0
        
        save_path = os.path.join(dst_data_dirs, os.path.dirname(_dir).split('/')[-1])
        csv_path = os.path.join(save_path, os.path.dirname(_dir).split('/')[-1] + '.csv')
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        try:
            for iter, df in enumerate(chunk_df(_dir, load_df_func=get_df)):
                # df = get_df(_dir)
                
                # drop Unnamed columns
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    
                # GENERATE METADATA
                if preprocessing_cfg['generate_metadata']:
                    print('- Generate metata')
                    meta = SingleTabMetadata(df)
                    metadata = meta.get_metadata(verbose=verbose)
                    
                # CLEAN DATA
                if preprocessing_cfg['clean_data']:
                    print('- Clean data')
                    
                    df, clean_info = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, 
                                                                                            min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                            pct_to_remove=preprocessing_cfg['percentage_to_remove'],
                                                                                            pct_to_remove_row=preprocessing_cfg['percentage_to_remove_row'],
                                                                                            return_info=True,
                                                                                            verbose=verbose)
                    clean_info_chunks.append(clean_info)
                    n_keep_rows += len(df)
                    
                    iterate_save_large_df(df, iter, csv_path)
                    
                    # print('DEBUG clean info: ', clean_info)
                    
                    # # skip this df if only 1 column
                    # if len(df.columns) <= min_cols_to_keep:
                    #     print('\t Skip because small number of columns: ', len(df.columns))
                    #     continue
                    
                    # # skip this df if number of rows <= 100
                    # if len(df) <= min_rows_to_keep:
                    #     print('\t Skip because small number of row: ', len(df))
                    #     continue

            clean_info = aggregate_clean_info(clean_info_chunks)
            n_keep_columns = len([k for k,v in clean_info.items() if v["keep"] == True])
            
            # skip this df if only 1 column
            if n_keep_columns <= min_cols_to_keep:
                print('\t Skip because small number of columns: ', len(df.columns))
                
                os.system(f"rm -rf {save_path}")
                continue
            
            # skip this df if number of rows <= 100
            if n_keep_rows <= min_rows_to_keep:
                print('\t Skip because small number of row: ', len(df))
                
                os.system(f"rm -rf {save_path}")
                continue
                
            # DENOTE ID COLUMN IN METADATA
            id_col_name = get_id_column_name(clean_info)
            
            if id_col_name is None:
                # id_col_name = list(clean_info.keys())[0]
                meta.metadata['primary_key'] = id_col_name
            
            else:
                meta.metadata['primary_key'] = id_col_name
                meta.metadata['columns'][id_col_name]['sdtype'] = "id"
                
            # SAVE
            # save dir setup
            from pathlib import Path
            # ds_dir = Path(_dir).stem
            # save_path = os.path.join(dst_data_dirs, ds_dir)
            
            # save metadata
            if preprocessing_cfg['generate_metadata']:
                prefix_metadta = ''
                meta.to_json(os.path.join(save_path, 'metadata.json'))
            
            # save cleaned df and clean info 
            if preprocessing_cfg['clean_data']:
                prefix_clean = ''
                # df.to_csv(os.path.join(save_path, ds_dir + '.csv'), index=False)
                # df.to_csv(os.path.join(save_path, prefix_clean + os.path.dirname(_dir).split('/')[-1] + '.csv'), index=False)
                json.dump(clean_info, open(os.path.join(save_path, 'clean_info.json'), 'w'))
        
        except pd.errors.EmptyDataError:
        #     # error_datasets.append(_dir)
        #     os.system(f"rm -rf {save_path}")
            continue
        
    # import pickle
    # pickle.dump(error_datasets, open(dst_data_dirs + '/error_datasets.pkl', 'wb'))
            
    return None