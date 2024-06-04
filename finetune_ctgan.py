import random
import json
from utils import *
from ctgan.synthesizers.ctgan import CustomCTGAN
import gc

# entry
# train

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs_ctganv2/pretraining_1e-4'
SAVE_PATH = 'rs_ctganv2/finetune_val_1e-4' # finetune_test_1e-4
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 
CHECKPOINT_EPOCH = None

TOTAL_EPOCHS = 100
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

training_hist = []

split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info[SET_NAME]
list_data_paths


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
FINETUNE_MODEL_CONFIG['discriminator_lr'] = DISCRIMINATOR_LR

# index = list_data_paths.index("50-startups-data")

for i, path in enumerate(list_data_paths):
    
    print(f'\t{path}')
    path = os.path.join(DATA_PATH, path)
    
    train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, init_transformer=False, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    transformer = get_transformer_v3(path)
    
    # train_data, val_data = load_tensor_data(path, 0.3, add_padding, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    # transformer = get_transformer(path)
    
    # load pretrained model
    finetune_model = CustomCTGAN(**FINETUNE_MODEL_CONFIG)
    finetune_model = load_gan_model_weights(finetune_model, PRETRAIN_PATH)
    
    # finetune
    ds_name = os.path.basename(path)
    
    finetune_model.fit(train_data, transformer, val_data, 
                       early_stopping=False,
                       checkpoint_epochs=CHECKPOINT_EPOCH, 
                       save_path=SAVE_PATH,
                       generator_name=str(ds_name) + '_generator',
                       discriminator_name=str(ds_name) + '_discriminator')
    
    
    training_hist = merge_training_hist(get_training_hist(finetune_model), ds_name, training_hist)
    
    # save
    generator_name = str(ds_name) + "_generator"
    discriminator_name = str(ds_name) + "_discriminator"
    
    save_gan_model_weights(finetune_model, SAVE_PATH, save_names=[generator_name, discriminator_name])
    save_training_history(training_hist, SAVE_PATH)
    
    gc.collect()
    
save_training_history(training_hist, SAVE_PATH)
