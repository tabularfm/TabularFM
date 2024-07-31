import yaml
import json
import numpy as np
from tabularfm.utils.processing import get_max_input_dim, get_max_n_categories, split_data

from tabularfm.ctgan.synthesizers.tvaev2 import CustomTVAE as STVAE
from tabularfm.ctgan.synthesizers.tvaev3 import CustomTVAE as STVAEM
from tabularfm.ctgan.synthesizers.tvae import CustomTVAE as OriTVAE
from tabularfm.ctgan.synthesizers.ctgan import CustomCTGAN as OriCTGAN
from tabularfm.be_great.great import CustomGReaT as OriGReaT

def _get_split_ratio(data_path, configs):
    
    split_ratio = configs['split_ratio']
    assert split_ratio is not None
    assert np.sum(split_ratio) == 1.0
    
    if len(split_ratio) == 2:
        return split_ratio[0], None
    elif len(split_ratio) == 3:
        return split_ratio[0], split_ratio[1]

    
def get_pretrain_paths(data_path, configs):
    
    if configs['split_path'] is None:
        pretrain_size, val_size = _get_split_ratio(data_path, configs)
        list_data_paths, _, _ = split_data(data_path, pretrain_size, val_size, random_state=configs['split_random_state'])
        return list_data_paths
    
    else:      
        split_path = configs['split_path']
        split_info = json.load(open(split_path, 'r'))
        list_data_paths = split_info['pretrain_paths']
        return list_data_paths
    
def get_finetune_paths(data_path, configs):
    
    if configs['split_path'] is None:
        pretrain_size, val_size = _get_split_ratio(data_path, configs)
        _, val_paths, test_paths = split_data(data_path, pretrain_size, val_size, random_state=configs['split_random_state'])
    
    else:
        split_path = configs['split_path']
        split_info = json.load(open(split_path, 'r'))
        val_paths, test_paths = split_info['val_paths'], split_info['test_paths']
        
    return val_paths, test_paths
    
def get_config(config_path):
    return yaml.safe_load(open(config_path))

def create_model_config(data_path, configs, model_type, config_type):
    if model_type in ['stvae', 'tvae', 'stvaem']:
        return _create_model_cfg_based_tvae(data_path, configs, config_type)

    if model_type in ['ctgan']:
        return _create_model_cfg_based_ctgan(data_path, configs, config_type)
    
    if model_type in ['great']:
        return _create_model_cfg_based_great(data_path, configs, config_type)
    
def create_model(model_type, model_config):
    if model_type == 'stvae':
        return _create_model_stvae(model_config)
    
    if model_type == 'stvaem':
        return _create_model_stvaem(model_config)
    
    if model_type == 'tvae':
        return _create_model_tvae(model_config)
    
    if model_type == 'ctgan':
        return _create_model_ctgan(model_config)
    
    if model_type == 'great':
        return _create_model_great(model_config)
    
    
def _create_model_cfg_based_tvae(data_path, configs, config_type):
    if config_type == 'pretrain':
        return {
            "input_dim": get_max_input_dim(data_path),
            "epochs": 1,
            "batch_size": configs['pretrain_cfg']['batch_size'],
            "lr": configs['pretrain_cfg']['lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "compress_dims": configs['model_cfg']['encoder_dims'],
            "decompress_dims": configs['model_cfg']['decoder_dims'],
            "verbose": configs['verbose']
        }   
    
    if config_type == 'finetune':
        return {
            "input_dim": get_max_input_dim(data_path),
            "epochs": configs['finetune_cfg']['epochs'],
            "batch_size": configs['finetune_cfg']['batch_size'],
            "lr": configs['finetune_cfg']['lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "compress_dims": configs['model_cfg']['encoder_dims'],
            "decompress_dims": configs['model_cfg']['decoder_dims'],
            "verbose": configs['verbose']
        }
    
    if config_type == 'fromscratch':
        return {
            "input_dim": None,
            "epochs": configs['fromscratch_cfg']['epochs'],
            "batch_size": configs['fromscratch_cfg']['batch_size'],
            "lr": configs['fromscratch_cfg']['lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "compress_dims": configs['model_cfg']['encoder_dims'],
            "decompress_dims": configs['model_cfg']['decoder_dims'],
            "verbose": configs['verbose']
        }
           
def _create_model_cfg_based_ctgan(data_path, configs, config_type):
    if config_type == 'pretrain':
        return {
            "input_dim": get_max_input_dim(data_path),
            "n_categories": get_max_n_categories(data_path),
            "epochs":  1,
            "batch_size": configs['pretrain_cfg']['batch_size'],
            "generator_lr": configs['pretrain_cfg']['generator_lr'],
            "discriminator_lr": configs['pretrain_cfg']['discriminator_lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "generator_dim": configs['model_cfg']['generator_dims'],
            "discriminator_dim": configs['model_cfg']['discriminator_dims'],
            "verbose": configs['verbose']
        }
    
    if config_type == 'finetune':
        return {
            "input_dim": get_max_input_dim(data_path),
            "n_categories": get_max_n_categories(data_path),
            "epochs":  configs['finetune_cfg']['epochs'],
            "batch_size": configs['finetune_cfg']['batch_size'],
            "generator_lr": configs['finetune_cfg']['generator_lr'],
            "discriminator_lr": configs['finetune_cfg']['discriminator_lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "generator_dim": configs['model_cfg']['generator_dims'],
            "discriminator_dim": configs['model_cfg']['discriminator_dims'],
            "verbose": configs['verbose']
        }
        
    if config_type == 'fromscratch':
        return {
            "input_dim": None,
            "n_categories": get_max_n_categories(data_path),
            "epochs":  configs['fromscratch_cfg']['epochs'],
            "batch_size": configs['fromscratch_cfg']['batch_size'],
            "generator_lr": configs['fromscratch_cfg']['generator_lr'],
            "discriminator_lr": configs['fromscratch_cfg']['discriminator_lr'],
            "embedding_dim": configs['model_cfg']['embedding_dim'],
            "generator_dim": configs['model_cfg']['generator_dims'],
            "discriminator_dim": configs['model_cfg']['discriminator_dims'],
            "verbose": configs['verbose']
        }
    
def _create_model_cfg_based_great(data_path, configs, config_type):
    if config_type == 'pretrain':
        return {
            "pretrained_llm": configs['model_cfg']['pretrained_llm'],
            "pretrained_tokenizer": configs['model_cfg']['pretrained_tokenizer'],
            "epochs": 1,
            "batch_size": configs['pretrain_cfg']['batch_size'],
            "model_max_length": configs['model_cfg']['token_max_length'],
            "init_from_scratch": configs['model_cfg']['init_from_scratch'],
            "verbose": configs['verbose']
        }
        
    if config_type == 'finetune':
        return {
            "pretrained_llm": configs['model_cfg']['pretrained_llm'],
            "pretrained_tokenizer": configs['model_cfg']['pretrained_tokenizer'],
            "epochs": configs['finetune_cfg']['epochs'],
            "batch_size": configs['finetune_cfg']['batch_size'],
            "model_max_length": configs['model_cfg']['token_max_length'],
            "verbose": configs['verbose']
        }
        
    if config_type == 'fromscratch':
        return {
            "pretrained_llm": configs['model_cfg']['pretrained_llm'],
            "pretrained_tokenizer": configs['model_cfg']['pretrained_tokenizer'],
            "epochs": configs['fromscratch_cfg']['epochs'],
            "batch_size": configs['fromscratch_cfg']['batch_size'],
            "model_max_length": configs['model_cfg']['token_max_length'],
            "verbose": configs['verbose']
        }
    
def _create_model_stvae(configs):
    return STVAE(**configs)

def _create_model_tvae(configs):
    return OriTVAE(**configs)

def _create_model_ctgan(configs):
    return OriCTGAN(**configs)

def _create_model_great(configs):
    return OriGReaT(**configs)

def _create_model_stvaem(configs):
    return STVAEM(**configs)