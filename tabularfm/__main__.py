"""CLI."""

import argparse
import os
import shutil
from tabularfm.pretrain import pretrain_model
from tabularfm.finetune import finetune_model
from tabularfm.trainfromscratch import train_from_scratch_model
from tabularfm.evaluate import evaluate_models

def _parse_args():
    parser = argparse.ArgumentParser(description='TabularFM Command Line Interface')
    
    # pipeline options
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, dest="pretrain", help="Run pretraining")
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, dest="finetune", help="Run finetuning")
    parser.add_argument('--fromscratch', action=argparse.BooleanOptionalAction, dest="fromscratch", help="Run training from scratch")
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, dest="evaluate", help="Run evaluation")
    
    # config
    parser.add_argument("-mt", "--model", type=str, default="stvae", dest="model", help="Path to the training data")
    parser.add_argument("-d", "--data", type=str, dest="data_path", help="Path to the directory of datasets")
    parser.add_argument("-s", "--save", type=str, dest="save_path", help="Save directory")
    parser.add_argument("-c", "--config", type=str, dest="config", help="Path to the configuration file")
    
    # kwargs
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, dest="resume", help="Whether to resume training or not")
    
    return parser.parse_args()

def setup_directory(path, resume):
    # if already exist but not resume, then replace
    if os.path.exists(path) and not resume:
        shutil.rmtree(path)
    
    # create save directory if not create
    if not os.path.exists(path) and not resume:
        os.mkdir(path)
        

def main():
    """CLI."""
    args = _parse_args()
    
    pretrain = args.pretrain
    finetune = args.finetune
    fromscratch = args.fromscratch
    evaluate = args.evaluate
    
    model_type = args.model
    data_path = args.data_path
    save_path = args.save_path
    config_path = args.config
    
    resume = args.resume
    
    setup_directory(save_path, resume)
    
    run_all_pipelines = not pretrain and not finetune and not fromscratch and not evaluate
    
    if pretrain or run_all_pipelines:
        print('*** PRETRAINING ***')
        pretrain_path = os.path.join(save_path, 'pretrain')
        setup_directory(pretrain_path, resume)
        pretrain_model(model_type, data_path, pretrain_path, config_path, resume)
    
    if finetune or run_all_pipelines:
        print('*** FINE-TUNNING ***')
        finetune_path = os.path.join(save_path, 'finetune')
        setup_directory(finetune_path, resume)
        finetune_model(model_type, data_path, finetune_path, config_path, pretrain_path, resume)
    
    if fromscratch or run_all_pipelines:
        print('*** TRAINING FROM SCRATCH ***')
        fromscratch_path = os.path.join(save_path, 'fromscratch')
        setup_directory(fromscratch_path, resume)
        train_from_scratch_model(model_type, data_path, fromscratch_path, config_path, resume)
    
    if evaluate or run_all_pipelines:
        print('*** EVALUATION ***')
        evaluate_path = os.path.join(save_path, 'evaluation')
        setup_directory(evaluate_path, resume)
        evaluate_models(model_type, data_path, finetune_path, fromscratch_path, evaluate_path, config_path, resume, True, True)
    
if __name__ == '__main__':
    main()