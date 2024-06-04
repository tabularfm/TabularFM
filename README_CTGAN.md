# Tabsyn v3 | CTGAN experiment

# Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r latest_requirements.txt
```

Make sure the available storage space at least 150GB

# Download and extract processed Kaggle datasets

Create directory to store data  
`mkdir data`

Download 1k2 Kaggle datasets at [GGDrive](https://drive.google.com/file/d/1VDAwgIp6Ts_rh3Vfm6dwOzrlFabiXkv8/view?usp=drive_link) into folder `data/`

Extract data  
`cd data`  
`unzip data_v3.zip`  
`cd ../`

The above command will extract datasets into `data/processed_dataset`


# Pretraining
Create directory to store experiment results  
`mkdir rs_ctganv2`  
`mkdir rs_ctganv2/pretraining_1e-4`

Train a model on large datasets (pretraining)  
`python pretrain_ctgan.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_ctganv2/pretraining_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 100 # save after every checkpoint epoch
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)
```

The result will be stored in the `SAVE_PATH`, the directory contains
* `generator_checkpoint_<n_epoch>.pt` and `discriminator_checkpoint_<n_epoch>.pt` for every `CHECKPOINT_EPOCH`
* `generator_weights.pt` and `discriminator_weights.pt`: final weights after completing the training
* `training_hist.csv`: training and validation loss of the training

# Finetuning
Create directory to store experiment results
* For evaluation set: `mkdir rs_ctganv2/finetune_val_1e-4`
* For test set: `mkdir rs_ctganv2/finetune_test_1e-4`


Finetune with a pretrained model per dataset (finetuning). The training procedure already applied early stopping.  
`python finetune_ctgan.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs_ctganv2/pretraining_1e-4'
SAVE_PATH = 'rs_ctganv2/finetune_val_1e-4' # finetune_test_1e-4
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # test_paths 
CHECKPOINT_EPOCH = None

TOTAL_EPOCHS = 100
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)
```

The above configuation for training evaluation set. To train test set
* switch `SAVE_PATH` from `'rs_ctganv2/finetune_val_1e-4'` to `'rs_ctganv2/finetune_test_1e-4'`
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`

The result will be stored in the `SAVE_PATH`, the directory contains
* `<dataset_name>_generator_checkpoint_<epoch>.pt` and `<dataset_name>_discriminator_checkpoint_<epoch.pt` are weights of finetuned model corresponding to the dataset at checkpoints.
* `<dataset_name>_generator.pt` and `<dataset_name>_discriminator.pt` are final weights of finetuned model corresponding to the dataset.
* `training_hist.csv`: training and validation loss of the training

# Single training per dataset

Create directory to store experiment results
* For evaluation set: `mkdir rs_ctganv2/single_val_1e-4`
* For test set: `mkdir rs_ctganv2/single_test_1e-4`

Train a model per dataset (single training)  
`python singletrain_ctgan.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_ctganv2/single_val_1e-4' # single_test_1e-4
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # test_paths
CHECKPOINT_EPOCH = None

TOTAL_EPOCHS = 100
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)
```

The above configuation for training evaluation set. To train test set
* switch `SAVE_PATH` from `'rs_ctganv2/single_val_1e-4'` to `'rs_ctganv2/single_test_1e-4'`
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`

# Evaluate the synthetic data
Generate data and compare the quality of synthetic data of finetuning vs. singletraining. The training procedure already applied early stopping.  
`python evaluate_syndata_ctgan.py`

Configuration
```python
DATA_PATH = 'data/processed_dataset'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # 'test_paths'
FINETUNE_PATH = 'rs_ctganv2/finetune_val_1e-4' # 'rs_ctganv2/finetune_test_1e-4' 
SINGLETRAIN_PATH = 'rs_ctganv2/single_val_1e-4' # 'rs_ctganv2/single_test_1e-4'
SCORE_SAVE_PATH = 'rs_ctganv2/scores_val.csv' # 'rs_ctganv2/scores_test.csv'
CHECKPOINT_EPOCH = None

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
GENERATOR_LR = 1.e-4
DISCRIMINATOR_LR = 1.e-4
EMBEDDING_DIM = 128
GENERATOR_DIMS = (512, 256, 256, 128)
DISCRIMINATOR_DIMS = (128, 256, 256, 512)
```

The `CHECKPOINT_EPOCH` is to evaluate the weight model at a specific checkpoint. To evaluate the final weight, set is to `None`

The above configuation for evaluate evaluation set. To evaluate test set
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`
* switch `FINETUNE_PATH` from `'rs_ctganv2/finetune_val_1e-4'` to `'rs_ctganv2/finetune_test_1e-4'`
* switch `SINGLETRAIN_PATH` from `'rs_ctganv2/single_val_1e-4'` to `'rs_ctganv2/single_test_1e-4'`
* switch `SCORE_SAVE_PATH` from `'rs_ctganv2/scores_val.csv'` to `'rs_ctganv2/scores_test.csv'`

# Run notebook to observe the report
Access the notebook `report_template.ipynb` to generate
* the training plot per dataset comparing between finetuning and singletraining (val set and test set)
* show the quality scores of synthetic data (as dataframe) between finetuning and singletraining (val set and test set)

Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH` to generate the training plot

Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres
