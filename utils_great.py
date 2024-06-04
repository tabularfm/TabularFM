
import os
import pandas as pd
from sdmetrics.reports.single_table import QualityReport
import json

def get_training_hist(trainer):
    return pd.DataFrame(trainer.state.log_history)

def merge_training_hist(new_hist, dataset_name, merged_hist):
    
    hist = new_hist.copy()
    hist['dataset'] = len(hist) * [str(dataset_name)]
    
    if len(merged_hist) == 0:
        merged_hist = hist
    
    else:
        merged_hist = pd.concat([merged_hist, hist])
        
    return merged_hist

def save_training_history(training_hist: pd.DataFrame, path: str):
    # index as orders to remember the sequence of training
    training_hist.index = range(len(training_hist))
    training_hist.to_csv(os.path.join(path, 'training_hist.csv'))
    
def save_model_weights(trainer, save_path):
    trainer.save_model()
    
def save_model_weights(trainer, path: str, save_name=None):
    
    if save_name is None:
        # trainer.save_model(os.path.join(path, 'weights.pt'))
        trainer.save_model(os.path.join(path, 'weights'))
    else:
        # trainer.save_model(os.path.join(path, f'{save_name}.pt'))
        trainer.save_model(os.path.join(path, f'{save_name}'))
        

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

def scoring(real_data, synthetic_data, metadata) -> QualityReport:
    report = QualityReport()
    
    report.generate(real_data, synthetic_data, metadata)
    
    return report

def get_metadata(path) -> dict:
    return json.load(open(path + '/metadata.json'))

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

def save_latest_pretrain_info_great(epoch, weight_path, path, filename=None):
    latest_training_info = {
        'epoch': epoch,
        'weights': weight_path,
    }
    
    filename = 'latest_training' if filename is None else filename
    json.dump(latest_training_info, open(os.path.join(path, f'{filename}.json'), 'w'))
    
def save_latest_ds_training_great(dataset_name, path, filename=None):
    latest_training_info = {
        'dataset': dataset_name,
    }
    
    filename = 'latest_training' if filename is None else filename
    json.dump(latest_training_info, open(os.path.join(path, f'{filename}.json'), 'w'))
    
def load_latest_training_info(path, filename=None):
    filename = 'latest_training' if filename is None else filename
    return json.load(open(os.path.join(path, f'{filename}.json'), 'r'))

def save_training_history(training_hist: pd.DataFrame, path: str, filename=None):
    # index as orders to remember the sequence of training
    training_hist.index = range(len(training_hist))
    
    filename = 'training_hist' if filename is None else filename
    training_hist.to_csv(os.path.join(path, f'{filename}.csv'))
