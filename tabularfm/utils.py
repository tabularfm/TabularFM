import yaml
import json
from ctgan.utils import get_max_input_dim

from ctgan.synthesizers.tvaev2 import CustomTVAE as CustomTVAEv2

def get_pretrain_paths(configs):
    split_path = configs['split_path']
    
    if split_path is None:
        # TODO
        pass
    
    else:
        split_info = json.load(open(split_path, 'r'))
        list_data_paths = split_info['pretrain_paths']
        
        return list_data_paths

def get_config(config_path):
    return yaml.safe_load(open(config_path))

def create_model_config(data_path, configs, model_type):
    if model_type == 'stvae':
        return _create_moddel_cfg_stvae(data_path, configs)
    
def create_model(model_type, model_config):
    if model_type == 'stvae':
        return _create_model_stvae(model_config)
    
def _create_moddel_cfg_stvae(data_path, configs):
    return {
        "input_dim": get_max_input_dim(data_path),
        "epochs": 1,
        "batch_size": configs['training_cfg']['batch_size'],
        "lr": configs['training_cfg']['lr'],
        "embedding_dim": configs['model_cfg']['embedding_dim'],
        "compress_dims": configs['model_cfg']['encoder_dims'],
        "decompress_dims": configs['model_cfg']['decoder_dims'],
        "verbose": configs['verbose']
    }
    
    
def _create_model_stvae(configs):
    return CustomTVAEv2(**configs)
