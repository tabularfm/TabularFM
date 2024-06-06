from utils import *
from tabularfm.ctgan.synthesizers.tvaev2 import CustomTVAE as CustomTVAEv2
import random

############# CONFIG #############

# DATA_PATH= 'data/processed_dataset'
DATA_PATH = '/mnt/hdd/jupyter/maindata/data/Quan/generative_models/tabsynv3/data/processed_dataset'
SAVE_PATH = 'rs_tvaev2_test/single_val_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

SINGLE_MODEL_CONFIG = {
    # "input_dim": None,
    "input_dim": get_max_input_dim(DATA_PATH),
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "embedding_dim": EMBEDDING_DIM,
    "compress_dims": ENCODERS_DIMS,
    "decompress_dims": DECODER_DIMS,
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
    single_model = CustomTVAEv2(**SINGLE_MODEL_CONFIG)
    
    ds_name = os.path.basename(path)
    single_model.fit(train_data, transformer, val_data,
                    early_stopping=True, 
                    checkpoint_epochs=None, 
                    save_path=SAVE_PATH,
                    encoder_name=str(ds_name) + '_encoder',
                    decoder_name=str(ds_name) + '_decoder')
    
    training_hist = merge_training_hist(get_training_hist(single_model), ds_name, training_hist)
    
    # save
    encoder_name = str(ds_name) + "_encoder"
    decoder_name = str(ds_name) + "_decoder"
    
    save_model_weights(single_model, SAVE_PATH, save_names=[encoder_name, decoder_name])
    save_training_history(training_hist, SAVE_PATH)
    
save_training_history(training_hist, SAVE_PATH)
