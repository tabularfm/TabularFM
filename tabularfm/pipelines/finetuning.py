import os
import gc
from tabularfm.utils.processing import load_tensor_data_v3, get_transformer_v3, add_padding, merge_training_hist, get_training_hist, save_latest_training_info, save_training_history, save_model_weights, get_df, get_colname_df, load_model_weights
from tabularfm.utils.cli import create_model
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from tabularfm.be_great.great_dataset import GReaTDataset
from tabularfm.ctgan.data_transformer import ColnameTransformer

def proceed_finetune(list_data_paths, configs, model_config, model_type, data_path, save_path, pretrain_path):
    if model_type in ['ctgan', 'tvae', 'stvae', 'stvaem']:
        _proceed_finetune_based_ctgan_tvae(list_data_paths, configs, model_config, model_type, data_path, save_path, pretrain_path)

    if model_type in ['great']:
        _proceed_finetune_based_great(list_data_paths, configs, model_config, model_type, data_path, save_path, pretrain_path)

def _proceed_finetune_based_ctgan_tvae(list_data_paths, configs, model_config, model_type, data_path, save_path, pretrain_path):
    DATA_PATH = data_path
    SAVE_PATH = save_path
    PRETRAIN_PATH = pretrain_path
    START_EPOCH = 0
    TOTAL_EPOCHS = configs['finetune_cfg']['epochs']
    # CHECKPOINT_EPOCH = configs['finetune_cfg']['checkpoint_n_epoch']
    EARLY_STOPPING = configs['finetune_cfg']['early_stopping'] if model_type != 'ctgan' else None
    
    # TODO: resume training
    training_hist = []
    
    if model_type == 'stvaem':
        PRETRAINED_LLM = configs['model_cfg']['pretrained_llm']
        colname_transformer = ColnameTransformer(pretrained_model=PRETRAINED_LLM)
        OPTIMIZE_COLUMN_NAME = configs['finetune_cfg']['optimize_signature']
    
    print(f'Total datasets {len(list_data_paths)}')
    
    for i, path in enumerate(list_data_paths):
        
        print(f'\t{path}')
        
        path = os.path.join(DATA_PATH, path)
        
        train_data, val_data = load_tensor_data_v3(path, 0.3, add_padding, 
                                                    init_transformer=False, 
                                                    **{'max_dim': model_config['input_dim']})
        
        transformer = get_transformer_v3(path)
        
        # load pretrained model
        model = create_model(model_type, model_config)
        model = load_model_weights(model_type, model, PRETRAIN_PATH, None)
        
        ds_name = os.path.basename(path)
        
        if model_type == 'stvaem':
            # column name transformer
            colname_texts = get_colname_df(path)
            colname_embeddings = colname_transformer.transform(colname_texts)
            colname_embeddings = colname_embeddings.detach().numpy().reshape(1, -1)
            
            model.fit(train_data, colname_embeddings, OPTIMIZE_COLUMN_NAME, transformer, val_data,
                    early_stopping=EARLY_STOPPING,
                    checkpoint_epochs=None,
                    save_path=SAVE_PATH,
                    encoder_name=f'encoder_{ds_name}',
                    decoder_name=f'decoder_{ds_name}')
            
            if not EARLY_STOPPING:
                save_model_weights(model_type, model, SAVE_PATH, suffix=ds_name)
                    
        else:
            if model_type in ['ctgan']:
                model.fit(train_data, transformer, val_data, 
                        early_stopping=False,
                        checkpoint_epochs=None,
                        save_path=SAVE_PATH,
                        generator_name=f'generator_{ds_name}',
                        discriminator_name=f'discriminator_{ds_name}')
                
                # save latested training info
                save_model_weights(model_type, model, SAVE_PATH, suffix=ds_name)
                
            elif model_type in ['tvae', 'stvae']:
                model.fit(train_data, transformer, val_data, 
                        early_stopping=EARLY_STOPPING,
                        checkpoint_epochs=None,
                        save_path=SAVE_PATH,
                        encoder_name=f'encoder_{ds_name}',
                        decoder_name=f'decoder_{ds_name}')
                
                if not EARLY_STOPPING:
                    save_model_weights(model_type, model, SAVE_PATH, suffix=ds_name)
        
        training_hist = merge_training_hist(get_training_hist(model_type, model), ds_name, training_hist)
        save_training_history(training_hist, SAVE_PATH)
        
        gc.collect()
           
        # TODO: support resume finetuning
        # save_latest_training_info(model_type, epoch, path, SAVE_PATH)
    
    # save training history at each epoch    
    save_training_history(training_hist, SAVE_PATH)
    
def _proceed_finetune_based_great(list_data_paths, configs, model_config, model_type, data_path, save_path, pretrain_path):
    PRETRAIN_PATH = pretrain_path
    DATA_PATH = data_path
    SAVE_PATH = save_path
    training_hist = []
    START_EPOCH = 0
    TOTAL_EPOCHS = configs['finetune_cfg']['epochs']
    # CHECKPOINT_EPOCH = configs['finetune_cfg']['checkpoint_n_epoch']
    EARLY_STOPPING = configs['finetune_cfg']['early_stopping']
        
    
    training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            save_strategy="no",
            learning_rate=configs['finetune_cfg']['lr'],
            num_train_epochs=configs['finetune_cfg']['epochs'],
            per_device_train_batch_size=configs['finetune_cfg']['batch_size'],
            per_device_eval_batch_size=configs['finetune_cfg']['batch_size'],
            logging_strategy='epoch',
            do_eval=True,
            evaluation_strategy='epoch',
        )
    
    for i, path in enumerate(list_data_paths):
        
        dataset_save_path = os.path.join(SAVE_PATH, path)
        path = os.path.join(DATA_PATH, path)
        
        if not EARLY_STOPPING:
            training_args = TrainingArguments(
                dataset_save_path,
                save_strategy='no',
                num_train_epochs=configs['finetune_cfg']['epochs'],
                per_device_train_batch_size=configs['finetune_cfg']['batch_size'],
                per_device_eval_batch_size=configs['finetune_cfg']['batch_size'],
                logging_strategy='epoch',
                do_eval=True,
                evaluation_strategy='epoch',
                # **self.train_hyperparameters,
            )
        else:
            training_args = TrainingArguments(
                output_dir=dataset_save_path,
                save_strategy='epoch',
                num_train_epochs=configs['finetune_cfg']['epochs'],
                per_device_train_batch_size=configs['finetune_cfg']['batch_size'],
                per_device_eval_batch_size=configs['finetune_cfg']['batch_size'],
                logging_strategy='epoch',
                do_eval=True,
                evaluation_strategy='epoch',
                metric_for_best_model = 'eval_loss',
                save_total_limit=1,
                load_best_model_at_end=True,
                # **self.train_hyperparameters,
            )
        
        
        df = get_df(path)
        
        n_rows, n_cols = len(df), len(df.columns)
        
        print(f'path: {path} | dataset: {path} | n_cols: {n_cols}, n_rows: {n_rows}')
        
        df, df_val = train_test_split(df, test_size=0.3, random_state=121)
        
        # modify the model_config to get the pretrained model
        pretrained_model_config = model_config.copy()
        pretrained_model_config['pretrained_llm'] = os.path.join(PRETRAIN_PATH, 'weights')
        
        # load pretrained model
        model = create_model(model_type, pretrained_model_config)
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
                        early_stopping=EARLY_STOPPING)
        
        ds_name = os.path.basename(path)
        
        training_hist = merge_training_hist(get_training_hist(model_type, great_trainer), ds_name, training_hist)
        save_training_history(training_hist, SAVE_PATH)    
        
        # TODO: save_latest_ds_training_great
        
        gc.collect()
        
# TODO: add param to save training hist with resume training