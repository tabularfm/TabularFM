import os
import gc
import random
from ctgan.utils import load_tensor_data_v3, get_transformer_v3, add_padding, merge_training_hist, get_training_hist, save_latest_training_info, save_training_history, save_model_weights, get_df, get_colname_df
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from be_great.great_dataset import GReaTDataset
from ctgan.data_transformer import ColnameTransformer

def proceed_pretrain(list_data_paths, configs, model_config, model, model_type, data_path, save_path):
    if model_type in ['ctgan', 'tvae', 'stvae', 'stvaem']:
        _proceed_pretrain_based_ctgan_tvae(list_data_paths, configs, model_config, model, model_type, data_path, save_path)

    if model_type in ['great']:
        _proceed_pretrain_based_great(list_data_paths, configs, model_config, model, model_type, data_path, save_path)

def _proceed_pretrain_based_ctgan_tvae(list_data_paths, configs, model_config, model, model_type, data_path, save_path):
    DATA_PATH = data_path
    SAVE_PATH = save_path
    START_EPOCH = 0
    TOTAL_EPOCHS = configs['training_cfg']['epochs']
    CHECKPOINT_EPOCH = configs['training_cfg']['checkpoint_n_epoch']
    PRETRAINED_LLM = configs['model_cfg']['pretrained_llm']
    OPTIMIZE_COLUMN_NAME = configs['training_cfg']['optimize_signature']
    
    # TODO: resume training
    training_hist = []
    
    if model_type == 'stvaem':
        colname_transformer = ColnameTransformer(pretrained_model=PRETRAINED_LLM)
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        print(f'EPOCH {epoch}')
        
        random.shuffle(list_data_paths)
        print(f'Epoch {epoch} with shuffled datasets {list_data_paths}')
        
        for i, path in enumerate(list_data_paths):
            
            print(f'\t{path}')
            
            path = os.path.join(DATA_PATH, path)
            
            train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, 
                                                       init_transformer=False, 
                                                       **{'max_dim': model_config['input_dim']})
            
            transformer = get_transformer_v3(path)
            
            if model_type == 'stvaem':
                # column name transformer
                colname_texts = get_colname_df(path)
                colname_embeddings = colname_transformer.transform(colname_texts)
                colname_embeddings = colname_embeddings.detach().numpy().reshape(1, -1)
                
                model.fit(train_data, colname_embeddings, OPTIMIZE_COLUMN_NAME, transformer, val_data)
            
            else:
                model.fit(train_data, transformer, val_data)
            
            ds_name = os.path.basename(path)
            training_hist = merge_training_hist(get_training_hist(model), ds_name, training_hist)
            
            gc.collect()
            
            # save latested training info
            save_model_weights(model_type, model, SAVE_PATH, suffix='temp')
            save_latest_training_info(model_type, epoch, path, SAVE_PATH)
            
        # save checkpoint
        if epoch >= CHECKPOINT_EPOCH and epoch % CHECKPOINT_EPOCH == 0:
            checkpoint = f'checkpoint_{epoch}'
            save_model_weights(model_type, model, SAVE_PATH, suffix=checkpoint)
        
        # save training history at each epoch    
        save_training_history(training_hist, SAVE_PATH)

    save_model_weights(model_type, model, SAVE_PATH)
    save_training_history(training_hist, SAVE_PATH)
    
def _proceed_pretrain_based_great(list_data_paths, configs, model_config, model, model_type, data_path, save_path):
    DATA_PATH = data_path
    SAVE_PATH = save_path
    training_hist = []
    START_EPOCH = 0
    TOTAL_EPOCHS = configs['training_cfg']['epochs']
    CHECKPOINT_EPOCH = configs['training_cfg']['checkpoint_n_epoch']
    
    training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            save_strategy="no",
            learning_rate=configs['training_cfg']['lr'],
            num_train_epochs=configs['training_cfg']['epochs'],
            per_device_train_batch_size=configs['training_cfg']['batch_size'],
            per_device_eval_batch_size=configs['training_cfg']['batch_size'],
            logging_strategy='epoch',
            do_eval=True,
            evaluation_strategy='epoch',
        )
    
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
                model.init_column_info(df)
            
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
            
            great_trainer = model.fit(great_ds_train, great_ds_val,
                            training_args,
                            resume_from_checkpoint=False,
                            early_stopping=False)
            
            ds_name = os.path.basename(path)
            
            training_hist = merge_training_hist(get_training_hist(great_trainer), ds_name, training_hist)
        
        # save latested training info
        weight_temp_name = f'temp_weights'
        save_model_weights(model_type, great_trainer, SAVE_PATH, weight_temp_name)
        save_latest_training_info(model_type, epoch, None, SAVE_PATH)   
        
        # save checkpoint
        if epoch >= CHECKPOINT_EPOCH and epoch % CHECKPOINT_EPOCH == 0:
            checkpoint = f'checkpoint_{epoch}'
            model_save_path = os.path.join(SAVE_PATH, f'weights_{checkpoint}')
            great_trainer.save_model(model_save_path)
        
        # save training history at each epoch    
        save_training_history(training_hist, SAVE_PATH)

    save_model_weights(model_type, great_trainer, SAVE_PATH)
    save_training_history(training_hist, SAVE_PATH)
    
    # TODO: add param to save training hist with resume training