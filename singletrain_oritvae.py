from utils import *
from tabularfm.ctgan.synthesizers.tvae import CustomTVAE
import random

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_oritvae/single_test_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'test_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############

SINGLE_MODEL_CONFIG = {
    "input_dim": None,
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
    
    # train_data, val_data = load_tensor_data_v2(path, 0.3, init_transformer=True)
    # transformer = get_transformer_v2(path)
    
    train_data, val_data = load_tensor_data_v3(path, 0.3)
    transformer = get_transformer_v3(path)
    
    SINGLE_MODEL_CONFIG["input_dim"] = train_data.shape[1]
    single_model = CustomTVAE(**SINGLE_MODEL_CONFIG)
    
    ds_name = os.path.basename(path)
    single_model.fit(train_data, transformer, val_data,
                    early_stopping=True, 
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
