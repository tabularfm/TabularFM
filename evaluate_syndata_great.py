import os
import json
from sklearn.model_selection import train_test_split
from be_great.great import CustomGReaT
from utils_great import *

DATA_PATH = 'data/processed_dataset'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths'
FINETUNE_PATH = 'rs/finetune_val'
SINGLETRAIN_PATH = 'rs/single_val'
SCORE_SAVE_PATH = 'rs/scores_val.csv'
RESUME_EVALUATION = False

TOTAL_EPOCHS = 500
BATCH_SIZE = 32 # paper
LR = 5.e-5 # paper
# EMBEDDING_DIM = 128
# ENCODERS_DIMS = (512, 256, 256, 128)
# DECODER_DIMS = (128, 256, 256, 512)

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

if RESUME_EVALUATION:
    latest_eval = load_latest_training_info(os.path.dirname(SCORE_SAVE_PATH), 'latest_evaluation')
    END_DATASET = latest_eval['dataset']
    
    ft_merged_df = pd.read_csv(os.path.join(FINETUNE_PATH, 'temp_scores.csv'), index_col=0)
    st_merged_df = pd.read_csv(os.path.join(SINGLETRAIN_PATH, 'temp_scores.csv'), index_col=0)
else:
    ft_merged_df = []
    st_merged_df = []

list_data_paths = json.load(open(SPLIT_INFO_PATH, 'r'))[SET_NAME]    
START_INDEX = list_data_paths.index(END_DATASET) + 1 if RESUME_EVALUATION else 0

for i, ds in enumerate(list_data_paths):
    
    if i < START_INDEX:
        continue
    
    # path
    path = os.path.join(DATA_PATH, ds)
    
    # load artifacts
    df = get_df(path)
    df, df_val = train_test_split(df, test_size=0.3, random_state=121)
    metadata = get_metadata(path)
    n_samples = len(df_val)
    
    # load weights, get the checkpoint folder as we only limit to 1 checkpoint
    finetune_ds_path = os.path.join(FINETUNE_PATH, ds)
    ft_weights_path = os.path.join(finetune_ds_path, os.listdir(finetune_ds_path)[0])
    ft_model = CustomGReaT(ft_weights_path)
    ft_model.init_column_info(df)
    
    singletrain_ds_path = os.path.join(SINGLETRAIN_PATH, ds)
    st_weights_path = os.path.join(singletrain_ds_path, os.listdir(singletrain_ds_path)[0])
    st_model = CustomGReaT(st_weights_path)
    st_model.init_column_info(df)
    
    # generate
    ft_syn_data = ft_model.sample(n_samples)
    st_syn_data = st_model.sample(n_samples)
    
    filtered_metadata = filter_metdata(metadata, st_syn_data.columns)
    
    # scoring
    ft_report = scoring(df_val, ft_syn_data, filtered_metadata)
    st_report = scoring(df_val, st_syn_data, filtered_metadata)
    
    ft_merged_df = add_score_df(ft_report, ds, ft_merged_df)
    st_merged_df = add_score_df(st_report, ds, st_merged_df)
    
    # save ft and st scores
    ft_merged_df.to_csv(os.path.join(FINETUNE_PATH, 'temp_scores.csv'))
    st_merged_df.to_csv(os.path.join(SINGLETRAIN_PATH, 'temp_scores.csv'))
    
    save_latest_ds_training_great(ds, os.path.dirname(SCORE_SAVE_PATH), 'latest_evaluation')

# merge scores
score_df = ft_merged_df.merge(st_merged_df, on='dataset', suffixes=['_ft', '_st'])
    
# add average row
average_data = score_df.drop(columns=['dataset']).mean().to_dict()
average_data['dataset'] = 'AVERAGE'
score_df.loc[len(score_df)] = average_data

score_df.to_csv(SCORE_SAVE_PATH)