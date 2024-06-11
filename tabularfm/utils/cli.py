import yaml
import json
from tabularfm.ctgan.processing import get_max_input_dim, get_max_n_categories

from ctgan.synthesizers.tvaev2 import CustomTVAE as STVAE
from ctgan.synthesizers.tvaev3 import CustomTVAE as STVAEM
from ctgan.synthesizers.tvae import CustomTVAE as OriTVAE
from ctgan.synthesizers.ctgan import CustomCTGAN as OriCTGAN
from be_great.great import CustomGReaT as OriGReaT

def get_pretrain_paths(configs):
    split_path = configs['split_path']
    
    if split_path is None:
        # TODO
        pass
    
    else:
        split_info = json.load(open(split_path, 'r'))
        list_data_paths = split_info['pretrain_paths']
        
        return list_data_paths
    
def get_finetune_paths(configs):
    split_path = configs['split_path']
    
    if split_path is None:
        # TODO
        pass
    
    else:
        split_info = json.load(open(split_path, 'r'))
        
        if configs['split_set'] in ['val', 'valdation', 'eval']:
            return split_info['val_paths']
        
        elif configs['split_set'] in ['test', 'testing']:
            return split_info['test_paths']

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