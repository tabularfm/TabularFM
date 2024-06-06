from utils import *
from tabularfm.ctgan.synthesizers.ctgan import CustomCTGAN
import random

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_ctganv2/single_val_1e-4' # single_test_1e-4
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # test_paths
CHECKPOINT_EPOCH = None # or None

TOTAL_EPOCHS = 100
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

SINGLE_MODEL_CONFIG = {
    "input_dim": None,
    "n_categories": None,
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "generator_lr": GENERATOR_LR,
    "discriminator_lr": DISCRIMINATOR_LR,
    "embedding_dim": EMBEDDING_DIM,
    "generator_dim": GENERATOR_DIMS,
    "discriminator_dim": DISCRIMINATOR_DIMS,
    "verbose": True
}

# entry
# train

training_hist = []

# list_data_paths = os.listdir(data_path)
split_info = json.load(open(SPLIT_INFO_PATH, 'r'))
list_data_paths = split_info[SET_NAME]

for i, path in enumerate(list_data_paths):
    
    print(f'\t{path}')
    path = os.path.join(DATA_PATH, path)
    
    train_data, val_data = load_tensor_data_v3(path, 0.3, init_transformer=False)
    transformer = get_transformer_v3(path)
    
    # train_data, val_data = load_tensor_data(path, 0.3)
    # transformer = get_transformer(path)
    
    SINGLE_MODEL_CONFIG["input_dim"] = train_data.shape[1]
    SINGLE_MODEL_CONFIG["n_categories"] = get_n_categories(path)
    single_model = CustomCTGAN(**SINGLE_MODEL_CONFIG)
    
    ds_name = os.path.basename(path)
    single_model.fit(train_data, transformer, val_data,
                    early_stopping=False, 
                    checkpoint_epochs=CHECKPOINT_EPOCH, 
                    save_path=SAVE_PATH,
                    generator_name=str(ds_name) + '_generator',
                    discriminator_name=str(ds_name) + '_discriminator')
    
    training_hist = merge_training_hist(get_training_hist(single_model), ds_name, training_hist)
    
    # save
    generator_name = str(ds_name) + "_generator"
    discriminator_name = str(ds_name) + "_discriminator"
    
    save_gan_model_weights(single_model, SAVE_PATH, save_names=[generator_name, discriminator_name])
    save_training_history(training_hist, SAVE_PATH)
    
save_training_history(training_hist, SAVE_PATH)
