import random
import json
from utils import *
from ctgan.synthesizers.tvae import CustomTVAE

# entry
# train

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs_oritvae/pretraining_1e-4'
SAVE_PATH = 'rs_oritvae/finetune_test_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'test_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############


# pretrain_encoder = 'encoder_checkpoint_450'
# pretrain_decoder = 'decoder_checkpoint_450'


training_hist = []

split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info[SET_NAME]
list_data_paths


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

FINETUNE_MODEL_CONFIG = MODEL_CONFIG.copy()
FINETUNE_MODEL_CONFIG['epochs'] = TOTAL_EPOCHS
FINETUNE_MODEL_CONFIG['lr'] = LR

for i, path in enumerate(list_data_paths):
    
    print(f'\t{path}')
    path = os.path.join(DATA_PATH, path)
    
    # train_data, val_data = load_tensor_data_v2(path, 0.3, add_padding, init_transformer=False, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    # transformer = get_transformer_v2(path)
    
    train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    transformer = get_transformer_v3(path)
    
    # load pretrained model
    finetune_model = CustomTVAE(**FINETUNE_MODEL_CONFIG)
    finetune_model = load_model_weights(finetune_model, PRETRAIN_PATH)
    
    # finetune
    ds_name = os.path.basename(path)
    
    finetune_model.fit(train_data, transformer, val_data, 
                       early_stopping=True,
                       save_path=SAVE_PATH,
                       encoder_name=str(ds_name) + '_encoder',
                       decoder_name=str(ds_name) + '_decoder')
    
    
    training_hist = merge_training_hist(get_training_hist(finetune_model), ds_name, training_hist)
    
    # save
    encoder_name = str(ds_name) + "_encoder"
    decoder_name = str(ds_name) + "_decoder"
    
    save_model_weights(finetune_model, SAVE_PATH, save_names=[encoder_name, decoder_name])
    save_training_history(training_hist, SAVE_PATH)
    
save_training_history(training_hist, SAVE_PATH)
