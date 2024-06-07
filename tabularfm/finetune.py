import os
import argparse
import shutil
from utils import get_config, create_model_config, create_model, get_finetune_paths
from pipelines.finetuning import proceed_finetune

def finetune_model(model_type, data_path, save_path, config_path, pretrain_path, resume):
    config_type = "finetune"
    
    configs = get_config(config_path)
    model_config = create_model_config(data_path, configs, model_type, config_type)
    model = create_model(model_type, model_config)
    list_data_paths = get_finetune_paths(configs)
    
    proceed_finetune(list_data_paths, configs, model_config, model, model_type, data_path, save_path, pretrain_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="stvae", dest="model", help="Path to the training data")
    parser.add_argument("-p", "--pretrain", type=str, default="stvae", dest="pretrain_path", help="Path to directory of pretrained model")
    parser.add_argument("-d", "--data", type=str, dest="data_path", help="Path to the directory of datasets")
    parser.add_argument("-s", "--save", type=str, dest="save_path", help="Save directory")
    parser.add_argument("-c", "--config", type=str, dest="config", help="Path to the configuration file")
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, dest="resume", help="Whether to resume training or not")

    args = parser.parse_args()
    
    # if already exist but not resume, then replace
    if os.path.exists(args.save_path) and not args.resume:
        shutil.rmtree(args.save_path)
    
    
    # create save directory if not create
    if not os.path.exists(args.save_path) and not args.resume:
        os.mkdir(args.save_path)
        
    # pretrain
    finetune_model(args.model, args.data_path, args.save_path, args.config, args.pretrain_path, args.resume)