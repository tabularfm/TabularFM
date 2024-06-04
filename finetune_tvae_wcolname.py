import random
import json
from ctgan.data_transformer import ColnameTransformer
from utils import *
from ctgan.synthesizers.tvaev3 import CustomTVAE as CustomTVAEv3

# entry
# train

############# CONFIG #############

DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs_tvaev2_wcolnameopt/pretraining_1e-4'
SAVE_PATH = 'rs_tvaev2_wcolnameopt/finetune_val_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 
OPTIMIZE_COLUMN_NAME = True

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

############# END CONFIG #############


# pretrain_encoder = 'encoder_checkpoint_430'
# pretrain_decoder = 'decoder_checkpoint_430'


training_hist = []

split_info = json.load(open(SPLIT_INFO_PATH, 'r'))

list_data_paths = split_info[SET_NAME]
list_data_paths


MODEL_CONFIG = {
    "input_dim": get_max_input_dim(DATA_PATH, colname_dim=768),
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

colname_transformer = ColnameTransformer()

for i, path in enumerate(list_data_paths):
    
    print(f'\t{path}')
    path = os.path.join(DATA_PATH, path)
    
    train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, init_transformer=False, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    transformer = get_transformer_v3(path)
    
    # train_data, val_data = load_tensor_data(path, 0.3, add_padding, **{'max_dim': FINETUNE_MODEL_CONFIG['input_dim']})
    # transformer = get_transformer(path)
    
    # column name transformer
    colname_texts = get_colname_df(path)
    colname_embeddings = colname_transformer.transform(colname_texts)
    colname_embeddings = colname_embeddings.detach().numpy().reshape(1, -1)
    
    # load pretrained model
    finetune_model = CustomTVAEv3(**FINETUNE_MODEL_CONFIG)
    finetune_model = load_model_weights(finetune_model, PRETRAIN_PATH)
    # finetune_model = load_model_weights(finetune_model, PRETRAIN_PATH, load_names=[pretrain_encoder, pretrain_decoder])
    
    # finetune
    ds_name = os.path.basename(path)
    
    finetune_model.fit(train_data, colname_embeddings, 
                       OPTIMIZE_COLUMN_NAME,
                       transformer, val_data, 
                       early_stopping=True,
                       checkpoint_epochs=None, 
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
