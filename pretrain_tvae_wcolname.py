# from ctgan.synthesizers.tvae import CustomTVAE
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random

from ctgan.synthesizers.tvaev3 import CustomTVAE as CustomTVAEv3

from ctgan.data_transformer import DataTransformer, ColnameTransformer
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
SAVE_PATH = 'rs_tvaev2_wcolnameopt/pretraining_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
OPTIMIZE_COLUMN_NAME = True
RESUME_TRAINING = False

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 10 # save after every checkpoint epoch
BATCH_SIZE = 512
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    "input_dim": get_max_input_dim(DATA_PATH, colname_dim=768),
    "epochs": 1,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "embedding_dim": EMBEDDING_DIM,
    "compress_dims": ENCODERS_DIMS,
    "decompress_dims": DECODER_DIMS,
    "verbose": True,
}

model = CustomTVAEv3(**MODEL_CONFIG)

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


# temp
# model = load_model_weights(model, SAVE_PATH, ['encoder_checkpoint_430', 'decoder_checkpoint_430'])

colname_transformer = ColnameTransformer()

training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info['pretrain_paths']
list_data_paths

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
# for epoch in range(431, TOTAL_EPOCHS):
    print(f'EPOCH {epoch}')
    
    random.shuffle(list_data_paths)
    print(f'Epoch {epoch} with shuffled datasets {list_data_paths}')
    
    for i, path in enumerate(list_data_paths):
        
        print(f'Epoch {epoch} | {path}')
        
        path = os.path.join(DATA_PATH, path)
        
        train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, init_transformer=False, **{'max_dim': MODEL_CONFIG['input_dim']})
        transformer = get_transformer_v3(path)
        
        # train_data, val_data = load_tensor_data(path, 0.3, add_padding, **{'max_dim': MODEL_CONFIG['input_dim']})
        # train_data, val_data = load_tensor_data(path, 0.3)
        # transformer = get_transformer(path)
        
        # column name transformer
        colname_texts = get_colname_df(path)
        colname_embeddings = colname_transformer.transform(colname_texts)
        colname_embeddings = colname_embeddings.detach().numpy().reshape(1, -1)
        
        # print('DEBUG NAN colname embedding: ', np.isnan(colname_embeddings).any())

        model.fit(train_data, colname_embeddings, OPTIMIZE_COLUMN_NAME,
                  transformer, val_data)
        
        ds_name = os.path.basename(path)
        training_hist = merge_training_hist(get_training_hist(model), ds_name, training_hist)
        
        gc.collect()
        
    # save latested training info
    generator_name = f'encoder_temp'
    discriminator_name = f'decoder_temp'
    save_model_weights(model, SAVE_PATH, save_names=[generator_name, discriminator_name])
    save_latest_training_info_tvae(epoch, path, generator_name, discriminator_name, SAVE_PATH)
        
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
