# from ctgan.synthesizers.tvae import CustomTVAE
import random

from ctgan.synthesizers.tvaev2 import CustomTVAE as CustomTVAEv2

from ctgan.data_transformer import DataTransformer
import pandas as pd
import pickle
import json
import os
import gc

import numpy as np

import matplotlib.pyplot as plt

from utils import *

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
# DATA_PATH = '/mnt/hdd/jupyter/maindata/data/Quan/generative_models/tabsynv3/data/ori_processed_dataset/processed_dataset/'
SAVE_PATH = 'rs_tvaev2_test/pretraining_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
RESUME_TRAINING = False

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 20 # save after every checkpoint epoch
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": 1,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "embedding_dim": EMBEDDING_DIM,
    "compress_dims": ENCODERS_DIMS,
    "decompress_dims": DECODER_DIMS,
    "verbose": True
}

print('DEBUG: input_dim: ', MODEL_CONFIG['input_dim'])

model = CustomTVAEv2(**MODEL_CONFIG)

if RESUME_TRAINING:
    latest_training = load_latest_training_info(SAVE_PATH)
    START_EPOCH = latest_training['epoch'] + 1
    RESUME_ENCODER = latest_training['encoder_weight']
    RESUME_DECODER = latest_training['decoder_weight']
    HISTORY_FILENAME = f'training_hist_e{START_EPOCH}'
    
    model = load_model_weights(model, 
                               path=SAVE_PATH, 
                               load_names=[RESUME_ENCODER, RESUME_DECODER])
    
else:
    START_EPOCH = 0
    HISTORY_FILENAME = 'training_hist'
    
training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info['pretrain_paths']
list_data_paths

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
    print(f'EPOCH {epoch}')
    
    random.shuffle(list_data_paths)
    print(f'Epoch {epoch} with shuffled datasets {list_data_paths}')
    
    for i, path in enumerate(list_data_paths):
        
        print(f'\t{path}')
        
        path = os.path.join(DATA_PATH, path)
        
        train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, init_transformer=False, **{'max_dim': MODEL_CONFIG['input_dim']})
        
        print('DEBUG train_data: ', train_data.shape)
        transformer = get_transformer_v3(path)
        
        # train_data, val_data = load_tensor_data(path, 0.3, add_padding, **{'max_dim': MODEL_CONFIG['input_dim']})
        # transformer = get_transformer(path)
        
        model.fit(train_data, transformer, val_data)
        
        ds_name = os.path.basename(path)
        training_hist = merge_training_hist(get_training_hist(model), ds_name, training_hist)
        
        gc.collect()
        
        # save latested training info
        encoder_name = f'encoder_temp'
        decoder_name = f'decoder_temp'
        save_model_weights(model, SAVE_PATH, save_names=[encoder_name, decoder_name])
        save_latest_training_info_tvae(epoch, path, encoder_name, decoder_name, SAVE_PATH)
        
    # save checkpoint
    if epoch >= CHECKPOINT_EPOCH and epoch % CHECKPOINT_EPOCH == 0:
        checkpoint = f'checkpoint_{epoch}'
        encoder_name = f'encoder_{checkpoint}'
        decoder_name = f'decoder_{checkpoint}'
        save_model_weights(model, SAVE_PATH, save_names=[encoder_name, decoder_name])
    
    # save training history at each epoch    
    save_training_history(training_hist, SAVE_PATH)

save_model_weights(model, SAVE_PATH)
save_training_history(training_hist, SAVE_PATH)