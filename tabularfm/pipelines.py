import os
import gc
import random
from ctgan.utils import load_tensor_data_v3, get_transformer_v3, add_padding, merge_training_hist, get_training_hist, save_latest_training_info, save_training_history, save_model_weights

def proceed_pretrain(list_data_paths, configs, model_config, model, model_type, data_path, save_path):
    DATA_PATH = data_path
    SAVE_PATH = save_path
    START_EPOCH = 0
    TOTAL_EPOCHS = configs['training_cfg']['epochs']
    CHECKPOINT_EPOCH = configs['training_cfg']['checkpoint_n_epoch']
    
    # TODO: resume training
    training_hist = []
    
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