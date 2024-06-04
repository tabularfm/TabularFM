# from ctgan.synthesizers.tvae import CustomTVAE
import random

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from be_great.great_dataset import GReaTDataset
from sklearn.model_selection import train_test_split
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
from be_great.great import CustomGReaT

import pandas as pd
import pickle
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from utils_great import *

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs/pretraining'
SPLIT_INFO_PATH = 'split_3sets.json'
RESUME_TRAINING = False

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 100 # save after every checkpoint epoch
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
# EMBEDDING_DIM = 128
# ENCODERS_DIMS = (512, 256, 256, 128)
# DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    # "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": 1,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    # "embedding_dim": EMBEDDING_DIM,
    # "compress_dims": ENCODERS_DIMS,
    # "decompress_dims": DECODER_DIMS,
    "verbose": True
}

if RESUME_TRAINING:
    latest_training = load_latest_training_info(SAVE_PATH)
    START_EPOCH = latest_training['epoch'] + 1
    WEIGHTS_PATH = latest_training['weights']
    RESUME_DECODER = latest_training['decoder_weight']
    HISTORY_FILENAME = f'training_hist_e{START_EPOCH}'
    
    great_model = CustomGReaT(WEIGHTS_PATH, 'distilgpt2',
                          SAVE_PATH,
                          MODEL_CONFIG['epochs'],
                          MODEL_CONFIG['batch_size'])
    
else:
    START_EPOCH = 0
    HISTORY_FILENAME = 'training_hist'
    
    great_model = CustomGReaT('distilgpt2', 'distilgpt2',
                          SAVE_PATH,
                          MODEL_CONFIG['epochs'],
                          MODEL_CONFIG['batch_size'])


training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            save_strategy="no",
            num_train_epochs=MODEL_CONFIG['epochs'],
            per_device_train_batch_size=MODEL_CONFIG['batch_size'],
            per_device_eval_batch_size=MODEL_CONFIG['batch_size'],
            logging_strategy='epoch',
            do_eval=True,
            evaluation_strategy='epoch',
        )

training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info['pretrain_paths']
list_data_paths

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
    
    random.shuffle(list_data_paths)
    print(f'Epoch {epoch} with shuffled datasets {list_data_paths}')
    
    for i, path in enumerate(list_data_paths):
        
        path = os.path.join(DATA_PATH, path)
        df = get_df(path)
        
        n_rows, n_cols = len(df), len(df.columns)
        
        print(f'Epoch: {epoch} | dataset: {path} | n_cols: {n_cols}, n_rows: {n_rows}')
        
        df, df_val = train_test_split(df, test_size=0.3, random_state=121)
        
        if epoch == 0:
            great_model.init_column_info(df)
        
        try:
            # train set
            great_ds_train = GReaTDataset.from_pandas(df)
            
            # val set
            great_ds_val = GReaTDataset.from_pandas(df_val)
        except:
            # this exception is expected to only deal with 1 dataset in gittables due to mixed dtypes
            # [CAUTION] below is only a workaround solution
            
            
            df = df.astype(str)
            df_val = df_val.astype(str)
            
            # train set
            great_ds_train = GReaTDataset.from_pandas(df)

            # val set
            great_ds_val = GReaTDataset.from_pandas(df_val)
            
            
        # adaptively configure batch size to avoid out-of-memory
        # if 10 < n_cols <= 20:
        #     training_args.per_device_train_batch_size = 16
        #     training_args.per_device_eval_batch_size = 16
        
        # if 20 < n_cols <= 30:
        #     training_args.per_device_train_batch_size = 8
        #     training_args.per_device_eval_batch_size = 8
            
        # if n_cols > 30:
        #     training_args.per_device_train_batch_size = 2
        #     training_args.per_device_eval_batch_size = 2
        
        great_trainer = great_model.fit(great_ds_train, great_ds_val,
                        training_args,
                        resume_from_checkpoint=False,
                        early_stopping=False)
        
        ds_name = os.path.basename(path)
        
        training_hist = merge_training_hist(get_training_hist(great_trainer), ds_name, training_hist)
        
                # break
                
        #     except:
        #         training_args.per_device_train_batch_size //=  2
        #         training_args.per_device_eval_batch_size //=  2
        #         print(f'*Out of memeory caught, reduce batch size to {training_args.per_device_train_batch_size}')
                
        # training_args.per_device_train_batch_size = MODEL_CONFIG['batch_size']
        # training_args.per_device_eval_batch_size = MODEL_CONFIG['batch_size']
    
    # save latested training info
    weight_temp_name = f'temp_weights'
    save_model_weights(great_trainer, SAVE_PATH, save_names=weight_temp_name)
    save_latest_pretrain_info_great(epoch, weight_temp_name, SAVE_PATH)   
    
    
    # save checkpoint
    if epoch >= CHECKPOINT_EPOCH and epoch % CHECKPOINT_EPOCH == 0:
        checkpoint = f'checkpoint_{epoch}'
        model_save_path = os.path.join(SAVE_PATH, f'weights_{checkpoint}')
        great_trainer.save_model(model_save_path)
    
    # save training history at each epoch    
    save_training_history(training_hist, SAVE_PATH, HISTORY_FILENAME)

save_model_weights(great_trainer, SAVE_PATH)
save_training_history(training_hist, SAVE_PATH, HISTORY_FILENAME)