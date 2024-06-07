import os
import json
from tabularfm.ctgan.synthesizers.ctgan import CustomCTGAN
from utils import *

DATA_PATH = 'data/processed_dataset'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths'
FINETUNE_PATH = 'rs_ctganv2/finetune_val_1e-4'
SINGLETRAIN_PATH = 'rs_ctganv2/single_val_1e-4'
SCORE_SAVE_PATH = 'rs_ctganv2/scores_val.csv'
CHECKPOINT_EPOCH = None

TOTAL_EPOCHS = 100
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)

MODEL_CONFIG = {
    "input_dim": get_max_input_dim(DATA_PATH),
    "n_categories": get_max_n_categories(DATA_PATH),
    "epochs": 1,
    "batch_size": BATCH_SIZE,
    "generator_lr": GENERATOR_LR,
    "discriminator_lr": DISCRIMINATOR_LR,
    "embedding_dim": EMBEDDING_DIM,
    "generator_dim": GENERATOR_DIMS,
    "discriminator_dim": DISCRIMINATOR_DIMS,
    "verbose": True
}


FINETUNE_MODEL_CONFIG = MODEL_CONFIG.copy()
FINETUNE_MODEL_CONFIG['epochs'] = TOTAL_EPOCHS
FINETUNE_MODEL_CONFIG['generator_lr'] = GENERATOR_LR
FINETUNE_MODEL_CONFIG["discriminator_lr"]= DISCRIMINATOR_LR

SINGLE_MODEL_CONFIG = MODEL_CONFIG.copy()
SINGLE_MODEL_CONFIG['epochs'] = TOTAL_EPOCHS
SINGLE_MODEL_CONFIG['generator_lr'] = GENERATOR_LR
SINGLE_MODEL_CONFIG["discriminator_lr"]= DISCRIMINATOR_LR

list_data_paths = json.load(open(SPLIT_INFO_PATH, 'r'))[SET_NAME]

ft_merged_df = []
st_merged_df = []

for ds in list_data_paths:
    
    # path
    path = os.path.join(DATA_PATH, ds)
    
    # load artifacts
    real_data = get_df(path)
    _, val_data = train_test_split(real_data, test_size=0.3, random_state=121)
    # transformer = get_transformer_v2(path)
    transformer = get_transformer_v3(path)
    metadata = get_metadata(path)
    n_samples = len(val_data)
    
    # find the optimal checkpoint
    # ft_loss, ft_val_loss = process_df_by_dataset(ft_training_hist, ds)
    # st_loss, st_val_loss = process_df_by_dataset(st_training_hist, ds)
    
    # ft_encoder_info, ft_decoder_info = get_checkpoint_info(ft_val_loss, ds)
    # st_encoder_info, st_decoder_info = get_checkpoint_info(st_val_loss, ds)
    
    checkpoint = '' if CHECKPOINT_EPOCH is None else '_checkpoint_' + str(CHECKPOINT_EPOCH)
    
    ft_generator_info, ft_discriminator_info = str(ds) + '_generator' + checkpoint, str(ds) + '_discriminator' + checkpoint
    st_generator_info, st_discriminator_info = str(ds) + '_generator' + checkpoint, str(ds) + '_discriminator' + checkpoint
    
    # load weights
    ft_model = CustomCTGAN(**FINETUNE_MODEL_CONFIG)
    ft_model = load_gan_model_weights(ft_model, FINETUNE_PATH, [ft_generator_info, ft_discriminator_info])
    ft_model.init_transformer(transformer)
    tensor_train_data, tensor_val_data = load_tensor_data_v3(path, 0.3, add_padding, init_transformer=False, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    ft_model.init_data_sampler(tensor_train_data, tensor_val_data)
    
    SINGLE_MODEL_CONFIG['input_dim'] = transformer.output_dimensions
    SINGLE_MODEL_CONFIG['n_categories'] = int(get_n_categories(path))
    st_model = CustomCTGAN(**SINGLE_MODEL_CONFIG)
    st_model = load_gan_model_weights(st_model, SINGLETRAIN_PATH, [st_generator_info, st_discriminator_info])
    st_model.init_transformer(transformer)
    tensor_train_data, tensor_val_data = load_tensor_data_v3(path, 0.3, init_transformer=False)
    st_model.init_data_sampler(tensor_train_data, tensor_val_data)
    
    # generate
    ft_syn_data = ft_model.sample(n=n_samples, transformer=transformer, condition_column=None, condition_value=None)
    st_syn_data = st_model.sample(n=n_samples, transformer=transformer, condition_column=None, condition_value=None)
    filtered_metadata = filter_metdata(metadata, st_syn_data.columns)
    
    # scoring
    ft_report = scoring(val_data, ft_syn_data, filtered_metadata)
    st_report = scoring(val_data, st_syn_data, filtered_metadata)
    
    ft_merged_df = add_score_df(ft_report, ds, ft_merged_df)
    st_merged_df = add_score_df(st_report, ds, st_merged_df)
    

# save ft and st scores
# ft_merged_df.to_csv(os.path.join(FINETUNE_PATH, 'scores.csv'))
# st_merged_df.to_csv(os.path.join(SINGLETRAIN_PATH, 'scores.csv'))

# merge scores
score_df = ft_merged_df.merge(st_merged_df, on='dataset', suffixes=['_ft', '_st'])
    
# add average row
average_data = score_df.drop(columns=['dataset']).mean().to_dict()
average_data['dataset'] = 'AVERAGE'
score_df.loc[len(score_df)] = average_data

score_df.to_csv(SCORE_SAVE_PATH)