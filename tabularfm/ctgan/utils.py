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
    
# def load_model_weights(model_type, model, path, load_names=[]):
#     if model_type in ['tvae', 'stvae', 'stvaem']:
#         return load_model_weights_based_tvae(model, path, load_names)
        
#     if model_type in ['ctgan']:
#         return load_model_weights_based_ctgan(model, path, load_names)
    
def load_model_weights(model_type, model, path, suffix):
    if model_type in ['tvae', 'stvae', 'stvaem']:
        return load_model_weights_based_tvae(model, path, suffix)
        
    if model_type in ['ctgan']:
        return load_model_weights_based_ctgan(model, path, suffix)
    
def load_model_weights_based_tvae(model: CustomTVAE, path: str, suffix=None) -> CustomTVAE:
    import torch
    
    if torch.cuda.is_available():
        if suffix is None:
            model.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_weights.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder_weights.pt')))
        else:
            model.encoder.load_state_dict(torch.load(os.path.join(path, f'encoder_{suffix}.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, f'decoder_{suffix}.pt')))
            
        return model
    
    else:
        if suffix is None:
            model.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_weights.pt')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder_weights.pt')))
        else:
            model.encoder.load_state_dict(torch.load(os.path.join(path, f'encoder_{suffix}.pt'), map_location=torch.device('cpu')))
            model.decoder.load_state_dict(torch.load(os.path.join(path, f'decoder_{suffix}.pt'), map_location=torch.device('cpu')))
    
        return model

def load_model_weights_based_ctgan(model: CustomCTGAN, path: str, suffix=None) -> CustomCTGAN:
    import torch
    
    if suffix is None:
        model._generator.load_state_dict(torch.load(os.path.join(path, 'generator_weights.pt')))
        model._discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator_weights.pt')))
    else:
        model._generator.load_state_dict(torch.load(os.path.join(path, f'generator_{suffix}.pt')))
        model._discriminator.load_state_dict(torch.load(os.path.join(path, f'discriminator_{suffix}.pt')))
    
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

def merge_column_shapes(ft_report, st_report, ds_name):
    df_shapes_ft = ft_report.get_details(property_name="Column Shapes").copy()
    df_shapes_st = st_report.get_details(property_name="Column Shapes").copy()

    df_shapes_ft.rename(columns={'Score': 'ft_score'}, inplace=True)
    df_shapes_ft.drop(columns=['Metric'], inplace=True)
    df_shapes_st.rename(columns={'Score': 'st_score'}, inplace=True)
    df_shapes_st.drop(columns=['Metric'], inplace=True)

    df_shapes = df_shapes_ft.merge(df_shapes_st, on='Column')
    df_shapes['dataset'] = ds_name
    
    return df_shapes


from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def visualize_colshape(model_type, split_set, df_shapes, save_path):
    # clean columns
    if 'Error_x' in df_shapes.columns or 'Error_y' in df_shapes.columns:
        na_indices = df_shapes[~df_shapes['Error_x'].isna() | (~df_shapes['Error_y'].isna())].index

        df_shapes.drop(index=na_indices, inplace=True)
        df_shapes.drop(columns=['Error_x', 'Error_y'], inplace=True)
        
    df_shapes['ft_st'] = df_shapes['ft_score'] - df_shapes['st_score']
    df_shapes['st_ft'] = df_shapes['st_score'] - df_shapes['ft_score']
    # df_shapes.sort_values(by=['diff'], ascending=False, inplace=True)
    
    # standardize column data
    df_shapes.replace('\n',' ', regex=True, inplace=True)
    
    # Remove columns with <string><number> or only <number>
    rules = r'Atr([0-9]*)(.*)|^([\s\d\d+\.\d+]+)$|Unnamed'
    filter = df_shapes['Column'].str.contains(rules)
    df_shapes = df_shapes[~filter]
    
    df_shapes.to_csv(os.path.join(save_path, f'colshape_{model_type}_{split_set}.csv'))

    col_names = df_shapes['Column'].to_list()
    freqs_ft = df_shapes['ft_st'].to_list()
    freqs_st = df_shapes['st_ft'].to_list()

    rs_dict_ft = {col_names[i]: freqs_ft[i] for i in range(len(col_names))}
    rs_dict_st = {col_names[i]: freqs_st[i] for i in range(len(col_names))}

    # FT > ST
    wordcloud_best = WordCloud(max_font_size=50, max_words=len(rs_dict_ft), background_color="white",
                        width=1280, height=800).generate_from_frequencies(rs_dict_ft)

    # plt.figure(figsize=(8, 6))

    fig = plt.gcf()
    plt.imshow(wordcloud_best, interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    fig.savefig(os.path.join(save_path, f'wc_{model_type}_{split_set}_best.png'), dpi=1000)

    # ST > FT
    wordcloud_worse = WordCloud(max_font_size=50, max_words=len(rs_dict_st), background_color="white",
                            colormap='Oranges',
                        width=1280, height=800).generate_from_frequencies(rs_dict_st)

    # plt.figure(figsize=(8, 6))

    fig = plt.gcf()
    plt.imshow(wordcloud_worse, interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    fig.savefig(os.path.join(save_path, f'wc_{model_type}_{split_set}_worst.png'), dpi=1000)
    
    
    
def merge_column_pairs(ft_report, st_report, ds_name):
    df_pairs_ft = ft_report.get_details(property_name="Column Pair Trends").copy()
    df_pairs_st = st_report.get_details(property_name="Column Pair Trends").copy()

    df_pairs_ft.rename(columns={'Score': 'ft_score'}, inplace=True)
    
    drop_columns = ['Metric', 'Real Correlation', 'Synthetic Correlation']
    if 'Error' in df_pairs_ft.columns:
        ft_drop_columns = drop_columns
        ft_drop_columns.append('Error')
        df_pairs_ft.drop(columns=ft_drop_columns, inplace=True)

    df_pairs_st.rename(columns={'Score': 'st_score'}, inplace=True)
    
    if 'Error' in df_pairs_st.columns:
        st_drop_columns = drop_columns
        st_drop_columns.append('Error')
        df_pairs_st.drop(columns=st_drop_columns, inplace=True)

    df_pairs = df_pairs_ft.merge(df_pairs_st, on=['Column 1', 'Column 2'])

    df_pairs['dataset'] = ds_name
    
    df_pairs = df_pairs[['Column 1', 'Column 2', 'ft_score', 'st_score', 'dataset']]
    df_pairs.dropna(subset=['ft_score', 'st_score'], inplace=True)
    
    return df_pairs
    

from pycirclize import Circos
from pycirclize.parser import Matrix
import pandas as pd

def visualize_colpair(model_type, split_set, df_pairs, save_path, top_k=30):
    
    df_pairs.replace('\n',' ', regex=True, inplace=True)

    df_pairs['ft_st'] = df_pairs['ft_score'] - df_pairs['st_score']
    df_pairs['st_ft'] = df_pairs['st_score'] - df_pairs['ft_score']

    # Remove columns with <string><number> or only <number>
    rules = r'Atr([0-9]*)(.*)|^([\s\d\d+\.\d+]+)$|Unnamed'
    filter = df_pairs['Column 1'].str.contains(rules)
    df_pairs = df_pairs[~filter]
    filter = df_pairs['Column 2'].str.contains(rules)
    df_pairs = df_pairs[~filter]
    
    df_pairs.to_csv(os.path.join(save_path, f'colpair_{model_type}_{split_set}.csv'))

    # Best pairs (FT > ST)
    data_best_pairs = df_pairs.sort_values(by='ft_st', ascending=False)[['Column 1', 'Column 2', 'ft_st']].iloc[:top_k].to_numpy()
    fromto_table_df = pd.DataFrame(data_best_pairs, columns=["from", "to", "value"])
    matrix_best_pairs = Matrix.parse_fromto_table(fromto_table_df)
 
    circos_best = Circos.initialize_from_matrix(
        matrix_best_pairs,
        space=3,
        cmap="viridis",
        # ticks_interval=5,
        label_kws=dict(size=8, r=110, orientation='vertical'),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )

    fig_best = circos_best.plotfig()
    fig_best.savefig(os.path.join(save_path, f'chord_{model_type}_{split_set}_best.png'), dpi=1000)
    
    # Worst pairs (FT > ST)
    data_worst_pairs = df_pairs.sort_values(by='st_ft', ascending=False)[['Column 1', 'Column 2', 'st_ft']].iloc[:top_k].to_numpy()
    fromto_table_df = pd.DataFrame(data_worst_pairs, columns=["from", "to", "value"])
    matrix_worst_pairs = Matrix.parse_fromto_table(fromto_table_df)
 
    circos_worst = Circos.initialize_from_matrix(
        matrix_worst_pairs,
        space=3,
        cmap="inferno",
        # ticks_interval=5,
        label_kws=dict(size=8, r=110, orientation='vertical'),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )

    fig_worst = circos_worst.plotfig()
    fig_worst.savefig(os.path.join(save_path, f'chord_{model_type}_{split_set}_worst.png'), dpi=1000)