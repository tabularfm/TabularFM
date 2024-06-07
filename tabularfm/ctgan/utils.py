# from ctgan.synthesizers.ctgan import CustomCTGAN
# from ctgan.synthesizers.tvae import CustomTVAE
from __future__ import annotations
from ctgan.data_transformer import DataTransformer
import pandas as pd
import csv
import pickle
import json
import os
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import numpy as np
from sdmetrics.reports.single_table import QualityReport
from ctgan.data_transformer import DataTransformerV2


def get_df(path, strict_mode=True) -> pd.DataFrame:
    csv_path = path + '/' + path.split('/')[-1] + '.csv'
    if not strict_mode:
        return pd.read_csv(csv_path, index_col=False)
    else:
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', encoding = 'utf-8')
        except:
            df = pd.read_csv(csv_path, on_bad_lines='skip', encoding = 'utf-8', lineterminator='\n', quoting=csv.QUOTE_NONE)
        
    return df

def get_colname_df(path):
    df = get_df(path)
    return df.columns.to_list()

def dump_transformer(path):
    metadata = json.load(open(path + '/metadata.json'))

    # get discrete cols
    discrete_cols = [k for k,v in metadata['columns'].items() if v['sdtype'] != 'numerical']

    # fit to transfomer
    df = get_df(path)
    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=discrete_cols)

    pickle.dump(transformer, open(path + '/transformer.pkl', 'wb'))
    
def dump_transformerv3(path, test_size=0.3, load_df_strict_mode=False):
    
    # load origina data
    df = get_df(path, load_df_strict_mode)
    data = df
    
    # split
    if test_size is not None:
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=121)
    else:
        train_data = data
        val_data = None
    
    transformer = DataTransformerV2()
    metadata = get_metadata(path)
    discrete_cols = [k for k,v in metadata['columns'].items() if v['sdtype'] != 'numerical']
    
    # fit only with train_data except one hot encoding should be fit on the whole data to prevent missing categories
    transformer.fit_wo_leakage(data, train_data.index, discrete_columns=discrete_cols)
    pickle.dump(transformer, open(path + '/transformer_v3.pkl', 'wb'))
    
    
# add padding
def add_padding(data, max_dim):
    assert len(data.shape) == 2
    
    return np.pad(data, ((0, 0), (0, max_dim - data.shape[1])), 'constant')

def torch_padding(data, max_dim):
    assert len(data.shape) == 2
    
    return F.pad(data, ((0, max_dim - data.shape[1], 0, 0)))
    
def get_transformer(path) -> DataTransformer:
    return pickle.load(open(path + '/transformer.pkl', 'rb'))

def get_transformer_v2(path) -> DataTransformer:
    return pickle.load(open(path + '/transformer_v2.pkl', 'rb'))

def get_transformer_v3(path) -> DataTransformerV2:
    return pickle.load(open(path + '/transformer_v3.pkl', 'rb'))


def get_metadata(path) -> dict:
    return json.load(open(path + '/metadata.json'))
    
def load_tensor_data(path, test_size=0.3, transform_func:callable=None, **kwargs):
    # load origina data
    df = get_df(path)

    transformer = get_transformer(path)
    
    # transform
    data = transformer.transform(df)
    
    # split
    if test_size is not None:
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=121)
    else:
        train_data = data
        val_data = None
        
    # transform func (e.g. add padding)
    if transform_func is not None:
        train_data = transform_func(train_data, **kwargs)
        
        if val_data is not None:
            val_data = transform_func(val_data, **kwargs)
    
    return train_data, val_data

def load_tensor_data_v2(path, test_size=0.3, transform_func:callable=None, init_transformer=False, **kwargs):
    # load origina data
    df = get_df(path)
    data = df
    
    # split
    if test_size is not None:
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=121)
    else:
        train_data = data
        val_data = None
        
    # transform
    if init_transformer:
        transformer = DataTransformer()
        metadata = get_metadata(path)
        discrete_cols = [k for k,v in metadata['columns'].items() if v['sdtype'] != 'numerical']
        transformer.fit(train_data, discrete_columns=discrete_cols)
        pickle.dump(transformer, open(path + '/transformer_v2.pkl', 'wb'))
    else:
        transformer = get_transformer_v2(path)
    
    train_data = transformer.transform(train_data)
    if test_size is not None:
        val_data = transformer.transform(val_data)
    
        
    # transform func (e.g. add padding)
    if transform_func is not None:
        train_data = transform_func(train_data, **kwargs)
        
        if val_data is not None:
            val_data = transform_func(val_data, **kwargs)
    
    return train_data, val_data

def load_tensor_data_v3(path, test_size=0.3, transform_func:callable=None, init_transformer=False, **kwargs):
    # load origina data
    df = get_df(path)
    data = df
    
    # split
    if test_size is not None:
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=121)
    else:
        train_data = data
        val_data = None
        
    transformer_path = os.path.join(path, 'transformer_v3.pkl')
    
    # transform
    if init_transformer or not os.path.exists(transformer_path):
        transformer = DataTransformerV2()
        metadata = get_metadata(path)
        discrete_cols = [k for k,v in metadata['columns'].items() if v['sdtype'] != 'numerical']
        
        # fit only with train_data except one hot encoding should be fit on the whole data to prevent missing categories
        transformer.fit_wo_leakage(data, train_data.index, discrete_columns=discrete_cols)
        pickle.dump(transformer, open(path + '/transformer_v3.pkl', 'wb'))
    else:
        transformer = get_transformer_v3(path)
    
    train_data = transformer.transform(train_data)
    if test_size is not None:
        val_data = transformer.transform(val_data)
    
        
    # transform func (e.g. add padding)
    if transform_func is not None:
        train_data = transform_func(train_data, **kwargs)
        
        if val_data is not None:
            val_data = transform_func(val_data, **kwargs)
    
    return train_data, val_data

def get_training_hist(model_type, obj) -> pd.DataFrame:
    if model_type in ['ctgan', 'tvae', 'stvae', 'stvaem']:
        return obj.loss_values
    
    if model_type in ['great']:
        return pd.DataFrame(obj.state.log_history)

# utils for training
def get_max_input_dim(data_path, colname_dim=None):
    paths = os.listdir(data_path)
    paths = [os.path.join(data_path, p) for p in paths]
    
    if colname_dim is None:
        return np.max([get_transformer_v3(path).output_dimensions for path in paths])
    
    return np.max([get_transformer_v3(path).output_dimensions + (colname_dim * len(get_transformer_v3(path)._column_transform_info_list)) for path in paths])

def get_max_n_categories(data_path):
    
    paths = os.listdir(data_path)
    paths = [os.path.join(data_path, p) for p in paths]
    max_n_categories = np.max([np.sum([k.output_dimensions for k in get_transformer_v3(path)._column_transform_info_list if k.column_type == 'discrete']) for path in paths])
    
    return int(max_n_categories)

def get_n_categories(path):
    return int(np.sum([k.output_dimensions for k in get_transformer(path)._column_transform_info_list if k.column_type == 'discrete']))
        

def merge_training_hist(new_hist, dataset_name, merged_hist):
    
    hist = new_hist.copy()
    hist['dataset'] = len(hist) * [str(dataset_name)]
    
    if len(merged_hist) == 0:
        merged_hist = hist
    
    else:
        merged_hist = pd.concat([merged_hist, hist])
        
    return merged_hist

def save_model_weights(model_type, obj, save_path, suffix=None):
    if model_type in ['tvae', 'stvae', 'stvaem']:
        if suffix is not None:
            encoder_name = f'encoder_{suffix}'
            decoder_name = f'decoder_{suffix}'
            save_names = [encoder_name, decoder_name]
        else:
            save_names = []
            
        save_model_weights_tvae(obj, save_path, save_names=save_names)
    
    if model_type in ['ctgan']:
        if suffix is not None:
            generator_name = f'generator_{suffix}'
            discriminator_name = f'discriminator_{suffix}'
            save_names = [generator_name, discriminator_name]
        else:
            save_names = []
            
        save_model_weights_gan(obj, save_path, save_names=save_names)
        
    if model_type in ['great']:
        save_model_weights_great(obj, save_path, suffix)

def save_model_weights_tvae(model: CustomTVAE, path: str, save_names=[]):
    import torch
    
    if len(save_names) == 0:
        torch.save(model.encoder.state_dict(), os.path.join(path, 'encoder_weights.pt'))
        torch.save(model.decoder.state_dict(), os.path.join(path, 'decoder_weights.pt'))
    else:
        torch.save(model.encoder.state_dict(), os.path.join(path, str(save_names[0]) + '.pt'))
        torch.save(model.decoder.state_dict(), os.path.join(path, str(save_names[1]) + '.pt'))
        
def save_model_weights_gan(model: CustomCTGAN, path: str, save_names=[]):
    import torch
    
    if len(save_names) == 0:
        torch.save(model._generator.state_dict(), os.path.join(path, 'generator_weights.pt'))
        torch.save(model._discriminator.state_dict(), os.path.join(path, 'discriminator_weights.pt'))
    else:
        torch.save(model._generator.state_dict(), os.path.join(path, str(save_names[0]) + '.pt'))
        torch.save(model._discriminator.state_dict(), os.path.join(path, str(save_names[1]) + '.pt'))
        
def save_model_weights_great(trainer, path: str, suffix=None):
    
    if suffix is None:
        trainer.save_model(os.path.join(path, 'weights'))
    else:
        trainer.save_model(os.path.join(path, f'{suffix}'))
    
def save_training_history(training_hist: pd.DataFrame, path: str, filename=None):
    # index as orders to remember the sequence of training
    training_hist.index = range(len(training_hist))
    
    filename = 'training_hist' if filename is None else filename
    training_hist.to_csv(os.path.join(path, f'{filename}.csv'))
    
def save_latest_training_info(model_type, epoch, step_path, save_path, suffix='temp'):
    
    if model_type in ['tvae', 'stvae', 'stvaem']:
        encoder_name = f'encoder_{suffix}'
        decoder_name = f'decoder_{suffix}'
        save_latest_training_info_tvae(epoch, step_path, encoder_name, decoder_name, save_path)
        
    if model_type in ['ctgan']:
        generator_name = f'generator_{suffix}'
        discriminator_name = f'discriminator_{suffix}'
        save_latest_training_info_gan(epoch, step_path, generator_name, discriminator_name, save_path)
        
    if model_type in ['great']:
        weight_path = 'temp_weights'
        save_latest_training_info_great(epoch, weight_path, save_path)
    
def save_latest_training_info_gan(epoch, step_path, generator_weight_path, discriminator_weight_path, path, filename=None):
    latest_training_info = {
        'epoch': epoch,
        'dataset': step_path,
        'generator_weight': generator_weight_path,
        'discriminator_weight': discriminator_weight_path,
    }
    
    filename = 'latest_training' if filename is None else filename
    json.dump(latest_training_info, open(os.path.join(path, f'{filename}.json'), 'w'))
    
def save_latest_training_info_tvae(epoch, step_path, encoder_weight_path, decoder_weight_path, path, filename=None):
    latest_training_info = {
        'epoch': epoch,
        'dataset': step_path,
        'encoder_weight': encoder_weight_path,
        'decoder_weight': decoder_weight_path,
    }
    
    filename = 'latest_training' if filename is None else filename
    json.dump(latest_training_info, open(os.path.join(path, f'{filename}.json'), 'w'))
    
def save_latest_training_info_great(epoch, weight_path, path, filename=None):
    latest_training_info = {
        'epoch': epoch,
        'weights': weight_path,
    }
    
    filename = 'latest_training' if filename is None else filename
    json.dump(latest_training_info, open(os.path.join(path, f'{filename}.json'), 'w'))
    
def load_latest_training_info(path, filename=None):
    filename = 'latest_training' if filename is None else filename
    return json.load(open(os.path.join(path, f'{filename}.json'), 'r'))
    
def load_model_weights(model_type, model, path, load_names=[]):
    if model_type in ['tvae', 'stvae', 'stvaem']:
        return load_model_weights_based_tvae(model, path, load_names)
        
    if model_type in ['ctgan']:
        return load_model_weights_based_ctgan(model, path, load_names)
    
def load_model_weights_based_tvae(model: CustomTVAE, path: str, load_names=[]) -> CustomTVAE:
    import torch
    
    if torch.cuda.is_available():
        if len(load_names) == 0:
            model.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_weights.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder_weights.pt')))
        else:
            model.encoder.load_state_dict(torch.load(os.path.join(path, str(load_names[0]) + '.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, str(load_names[1]) + '.pt')))
            
        return model
    
    else:
        if len(load_names) == 0:
            model.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_weights.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder_weights.pt')))
        else:
            model.encoder.load_state_dict(torch.load(os.path.join(path, str(load_names[0]) + '.pt'), map_location=torch.device('cpu')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, str(load_names[1]) + '.pt'), map_location=torch.device('cpu')))
    
        return model

def load_model_weights_based_ctgan(model: CustomCTGAN, path: str, load_names=[]) -> CustomCTGAN:
    import torch
    
    if len(load_names) == 0:
        model._generator.load_state_dict(torch.load(os.path.join(path, 'generator_weights.pt')))
        model._discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator_weights.pt')))
    else:
        model._generator.load_state_dict(torch.load(os.path.join(path, str(load_names[0]) + '.pt')))
        model._discriminator.load_state_dict(torch.load(os.path.join(path, str(load_names[1]) + '.pt')))
    
    return model
    

# utils

def scoring(real_data, synthetic_data, metadata) -> QualityReport:
    report = QualityReport()
    
    report.generate(real_data, synthetic_data, metadata)
    
    return report

def modify_metadata(metadata):
    new_metadata = {}

    columns_dict = {}
    for k,v in metadata['fields'].items():
        columns_dict[k] = {}
        for i,j in v.items():
            if i == 'type':
                columns_dict[k]['sdtype'] = j
            # elif i == 'subtype':
            #     columns_dict[k][]
            else:
                columns_dict[k][i] = j
                
    new_metadata['primary_key'] = metadata['primary_key']
    new_metadata['columns'] = columns_dict
    
    return new_metadata

def filter_metdata(metadata, columns):
    
    new_metadata = {}
    new_metadata['primary_key'] = metadata['primary_key']

    columns_dict = {}
    for k,v in metadata['columns'].items():
        
        if k not in columns:
            continue
        
        columns_dict[k] = {}
        for i,j in v.items():
            columns_dict[k][i] = j
                
    new_metadata['primary_key'] = metadata['primary_key']
    new_metadata['columns'] = columns_dict

    return new_metadata


def add_score_df(new_report: QualityReport, dataset, merged_df):
    new_df = pd.DataFrame([{
        'dataset': str(dataset),
        'column_shapes': new_report.get_properties().iloc[0, 1],
        'column_pair_trends': new_report.get_properties().iloc[1,1],
        'overall_score': new_report.get_score(),
    }]
    )
    
    if len(merged_df) == 0:
        merged_df = new_df
        
    else:
        merged_df = pd.concat([merged_df, new_df])
        
    return merged_df

def save_scores_df(scores_df: pd.DataFrame, path: str):
    # index to remeber the sequence of testing
    scores_df.index = range(len(scores_df))
    scores_df.to_csv(os.path.join(path, 'scores.csv'))
    
    
def process_df_by_dataset(df, dataset_name):
    grouped_df = df[df['dataset'] == dataset_name].groupby('epoch')[['val_loss', 'loss']].mean()
    
    loss = grouped_df['loss'].to_numpy()
    val_loss = grouped_df['val_loss'].to_numpy()
    
    assert len(loss) == len(df['epoch'].unique())
    assert len(val_loss) == len(df['epoch'].unique())
    
    return loss, val_loss