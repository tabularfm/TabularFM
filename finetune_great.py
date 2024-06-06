# from ctgan.synthesizers.tvae import CustomTVAE
import random

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, EarlyStoppingCallback
from tabularfm.be_great.great_dataset import GReaTDataset
from sklearn.model_selection import train_test_split
from tabularfm.be_great.great_dataset import GReaTDataset, GReaTDataCollator
from tabularfm.be_great.great_trainer import GReaTTrainer
from tabularfm.be_great.great import CustomGReaT

import pandas as pd
import pickle
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from utils_great import *

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs/pretraining/weights'
SAVE_PATH = 'rs/finetune_val'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 
RESUME_TRAINING = True

TOTAL_EPOCHS = 500
# CHECKPOINT_EPOCH = 25 # save after every checkpoint epoch
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
# EMBEDDING_DIM = 128
# ENCODERS_DIMS = (512, 256, 256, 128)
# DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

MODEL_CONFIG = {
    # "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    # "embedding_dim": EMBEDDING_DIM,
    # "compress_dims": ENCODERS_DIMS,
    # "decompress_dims": DECODER_DIMS,
    "verbose": True
}

if RESUME_TRAINING:
    latest_training = load_latest_training_info(SAVE_PATH)
    END_DATASET = latest_training['dataset']
    training_hist = pd.read_csv(os.path.join(SAVE_PATH, 'training_hist.csv'), index_col=0)
else:
    training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info[SET_NAME]

START_INDEX = list_data_paths.index(END_DATASET) + 1 if RESUME_TRAINING else 0

for i, path in enumerate(list_data_paths):
    
    if i < START_INDEX:
        continue
    
    dataset_save_path = os.path.join(SAVE_PATH, path)
    
    path = os.path.join(DATA_PATH, path)
    df = get_df(path)
    n_rows, n_cols = len(df), len(df.columns)
        
    print(f'path: {path} | dataset: {path} | n_cols: {n_cols}, n_rows: {n_rows}')
    
    pretrained_great_model = CustomGReaT(PRETRAIN_PATH,
                              'distilgpt2',
                              dataset_save_path,
                              MODEL_CONFIG['epochs'],
                              MODEL_CONFIG['batch_size'])
    
    df, df_val = train_test_split(df, test_size=0.3, random_state=121)

    pretrained_great_model.init_column_info(df)
    
    # train set
    great_ds_train = GReaTDataset.from_pandas(df)

    # val set
    great_ds_val = GReaTDataset.from_pandas(df_val)
    
    # if 10 < n_cols <= 20:
    #     MODEL_CONFIG['batch_size'] = 16
    #     MODEL_CONFIG['batch_size'] = 16
    
    # if 20 < n_cols <= 30:
    #     MODEL_CONFIG['batch_size'] = 8
    #     MODEL_CONFIG['batch_size'] = 8
        
    # if n_cols > 30:
    #     MODEL_CONFIG['batch_size'] = 2
    #     MODEL_CONFIG['batch_size'] = 2
    
    finetune_trainer = pretrained_great_model.fit(great_ds_train, 
                                               great_ds_val,
                                               training_args=None,
                                               great_trainer=None,
                                               early_stopping=True)
    
    ds_name = os.path.basename(path)

    training_hist = merge_training_hist(get_training_hist(finetune_trainer), ds_name, training_hist)
    
    # MODEL_CONFIG['batch_size'] = BATCH_SIZE
    
    save_training_history(training_hist, SAVE_PATH)    
    save_latest_ds_training_great(ds_name, SAVE_PATH)
    
    
    