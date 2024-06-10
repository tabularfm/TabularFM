import os
from ctgan.utils import get_df, get_transformer_v3, get_metadata, load_model_weights, filter_metdata, scoring, add_score_df, get_colname_df, load_tensor_data_v3, add_padding, merge_column_shapes, merge_column_pairs
from ctgan.data_transformer import ColnameTransformer
from sklearn.model_selection import train_test_split
from utils import create_model
import pandas as pd


def proceed_scoring(list_data_paths, configs, model_config_finetune, model_config_fromscratch, model_type, data_path, finetune_path, fromscratch_path, save_path):
    if model_type in ['ctgan', 'tvae', 'stvae', 'stvaem']:
        return _proceed_finetune_based_ctgan_tvae(list_data_paths, configs, model_config_finetune, model_config_fromscratch, model_type, data_path, finetune_path, fromscratch_path, save_path)
        
    if model_type in ['great']:
        return _proceed_finetune_based_great(list_data_paths, configs, model_config_finetune, model_config_fromscratch, model_type, data_path, finetune_path, fromscratch_path, save_path)
        
def _proceed_finetune_based_ctgan_tvae(list_data_paths, configs, model_config_finetune, model_config_fromscratch, model_type, data_path, finetune_path, fromscratch_path, save_path):
    DATA_PATH = data_path
    SCORE_SAVE_PATH = save_path
    
    ft_merged_df = []
    st_merged_df = []
    
    list_df_shapes = []
    list_df_pairs = []

    for ds in list_data_paths:
        # path
        path = os.path.join(DATA_PATH, ds)
        
        # load artifacts
        real_data = get_df(path)
        
        # get real data
        _, val_data = train_test_split(real_data, test_size=0.3, random_state=121)
        
        # transform data
        transformer = get_transformer_v3(path)
        metadata = get_metadata(path)
        n_samples = len(val_data)
        
        # load weights
        ft_model = create_model(model_type, model_config_finetune)
        ft_model = load_model_weights(model_type, ft_model, finetune_path, suffix=ds)
        
        if model_type in ['ctgan', 'tvae', 'stvae']:
            model_config_fromscratch['input_dim'] = transformer.output_dimensions
            
        elif model_type in ['stvaem']:
            colname_texts = get_colname_df(path)
            colname_transformer = ColnameTransformer(configs['model_cfg']['pretrained_llm'])
            colname_embeddings = colname_transformer.transform(colname_texts)
            colname_embeddings = colname_embeddings.detach().numpy().reshape(1, -1)
            
            _, tensor_val_data = load_tensor_data_v3(path, 0.3)
            model_config_fromscratch['input_dim'] = tensor_val_data.shape[1] + (len(colname_texts) * 768)
        
        st_model = create_model(model_type, model_config_fromscratch)
        st_model = load_model_weights(model_type, st_model, fromscratch_path, suffix=ds)
        
        # ctgan model needs to init transform and data sampler before generating data
        if model_type in ['ctgan']:
            
            # init transformer
            ft_model.init_transformer(transformer)
            st_model.init_transformer(transformer)
            
            # init sampler
            tensor_train_data, tensor_val_data = load_tensor_data_v3(path, 0.3)
            st_model.init_data_sampler(tensor_train_data, tensor_val_data)
            
            tensor_train_data, tensor_val_data = load_tensor_data_v3(path, 0.3, add_padding, **{'max_dim': model_config_finetune['input_dim']})
            ft_model.init_data_sampler(tensor_train_data, tensor_val_data)
        
        # generate
        ft_syn_data = ft_model.sample(n_samples, transformer)
        st_syn_data = st_model.sample(n_samples, transformer)
        filtered_metadata = filter_metdata(metadata, st_syn_data.columns)
        
        # scoring
        ft_report = scoring(val_data, ft_syn_data, filtered_metadata)
        st_report = scoring(val_data, st_syn_data, filtered_metadata)
        
        ft_merged_df = add_score_df(ft_report, ds, ft_merged_df)
        st_merged_df = add_score_df(st_report, ds, st_merged_df)
        
        list_df_shapes.append(merge_column_shapes(ft_report, st_report, ds))
        list_df_pairs.append(merge_column_pairs(ft_report, st_report, ds))
        
    # merge scores
    score_df = ft_merged_df.merge(st_merged_df, on='dataset', suffixes=['_ft', '_st'])
        
    # add average row
    average_data = score_df.drop(columns=['dataset']).mean().to_dict()
    average_data['dataset'] = 'AVERAGE'
    score_df.loc[len(score_df)] = average_data

    score_df.to_csv(SCORE_SAVE_PATH)
    
    
    df_shapes = pd.concat(list_df_shapes) if len(list_df_shapes) > 0 else None
    df_pairs = pd.concat(list_df_pairs) if len(list_df_pairs) > 0 else None
    
    return df_shapes, df_pairs
    
def _proceed_finetune_based_great(list_data_paths, configs, model_config_finetune, model_config_fromscratch, model_type, data_path, finetune_path, fromscratch_path, save_path, colshape_viz, colpair_viz):
    DATA_PATH = data_path
    SCORE_SAVE_PATH = save_path
    
    ft_merged_df = []
    st_merged_df = []
    
    for i, ds in enumerate(list_data_paths):
        # path
        path = os.path.join(DATA_PATH, ds)
        
         # load artifacts
        df = get_df(path)
        df, df_val = train_test_split(df, test_size=0.3, random_state=121)
        metadata = get_metadata(path)
        n_samples = len(df_val)
        
        # load weights by getting the checkpoint folder as we only limit to 1 checkpoint
        finetune_ds_path = os.path.join(finetune_path, ds)
        ft_weights_path = os.path.join(finetune_ds_path, os.listdir(finetune_ds_path)[0])
        model_config_finetune['pretrained_llm'] = os.path.join(ft_weights_path)
        ft_model = create_model(model_type, model_config_finetune)
        ft_model.init_column_info(df)
        
        singletrain_ds_path = os.path.join(fromscratch_path, ds)
        st_weights_path = os.path.join(singletrain_ds_path, os.listdir(singletrain_ds_path)[0])
        model_config_fromscratch['pretrained_llm'] = os.path.join(st_weights_path)
        st_model = create_model(model_type, model_config_fromscratch)
        st_model.init_column_info(df)
        
        # generate
        ft_syn_data = ft_model.sample(n_samples)
        st_syn_data = st_model.sample(n_samples)
        
        filtered_metadata = filter_metdata(metadata, st_syn_data.columns)
        
        # scoring
        ft_report = scoring(df_val, ft_syn_data, filtered_metadata)
        st_report = scoring(df_val, st_syn_data, filtered_metadata)
        
        ft_merged_df = add_score_df(ft_report, ds, ft_merged_df)
        st_merged_df = add_score_df(st_report, ds, st_merged_df)
        
    # merge scores
    score_df = ft_merged_df.merge(st_merged_df, on='dataset', suffixes=['_ft', '_st'])
        
    # add average row
    average_data = score_df.drop(columns=['dataset']).mean().to_dict()
    average_data['dataset'] = 'AVERAGE'
    score_df.loc[len(score_df)] = average_data

    score_df.to_csv(SCORE_SAVE_PATH)