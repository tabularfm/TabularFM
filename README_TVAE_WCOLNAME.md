# Tabsyn v3 | TVAE experiment with column name emebddings but without optimization (TVAE_wcolname_woopt)

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

Download 1k2 Kaggle datasets at [GGDrive](https://drive.google.com/file/d/1oIcTzLupszhIjy6VUG7WK5l5vOYCXOEi/view?usp=sharing) into folder `data/`

Extract data  
`cd data`  
`unzip data_v5.zip`  
`cd ../`

The above command will extract datasets into `data/processed_dataset`


# Pretraining
Create directory to store experiment results  
`mkdir rs_tvaev2_wcolname_woopt`  
`mkdir rs_tvaev2_wcolname_woopt/pretraining_1e-4`

Train a model on large datasets (pretraining)  
`python pretrain_tvae_wcolname_woopt.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_tvaev2_wcolname_woopt/pretraining_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
OPTIMIZE_COLUMN_NAME = False # keep this config as False
RESUME_TRAINING = False # set True to resume training

TOTAL_EPOCHS = 500
CHECKPOINT_EPOCH = 100 # save after every checkpoint epoch
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

# Finetuning
Create directory to store experiment results
* For evaluation set: `mkdir rs_tvaev2_wcolname_woopt/finetune_val_1e-4`
* For test set: `mkdir rs_tvaev2_wcolname_woopt/finetune_test_1e-4`


Finetune with a pretrained model per dataset (finetuning). The training procedure already applied early stopping.  
`python finetune_tvae_wcolname_woopt.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs_tvaev2_wcolname_woopt/pretraining_1e-4'
SAVE_PATH = 'rs_tvaev2_wcolname_woopt/finetune_val_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 
OPTIMIZE_COLUMN_NAME = False

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

The above configuation for training evaluation set. To train test set
* switch `SAVE_PATH` from `'rs_tvaev2_wcolname_woopt/finetune_val_1e-4` to `'rs_tvaev2_wcolname_woopt/finetune_test_1e-4'`
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`


# Single training per dataset

Create directory to store experiment results
* For evaluation set: `mkdir rs_tvaev2_wcolname_woopt/single_val_1e-4`
* For test set: `mkdir rs_tvaev2_wcolname_woopt/single_test_1e-4`

Train a model per dataset (single training)  
`python singletrain_tvae_wcolname_woopt.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs_tvaev2_wcolname_woopt/single_val_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 
OPTIMIZE_COLUMN_NAME = False

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

The above configuation for training evaluation set. To train test set
* switch `SAVE_PATH` from `'rs_tvaev2_wcolname_woopt/single_val_1e-4'` to `'rs_tvaev2_wcolname_woopt/single_test_1e-4'`
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`

# Evaluate the synthetic data
Generate data and compare the quality of synthetic data of finetuning vs. singletraining. The training procedure already applied early stopping.  
`python evaluate_syndata_tvae_wcolname_woopt.py`

Configuration
```python
DATA_PATH = 'data/processed_dataset'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths'
FINETUNE_PATH = 'rs_tvaev2_wcolname_woopt/finetune_val_1e-4'
SINGLETRAIN_PATH = 'rs_tvaev2_wcolname_woopt/single_val_1e-4'
SCORE_SAVE_PATH = 'rs_tvaev2_wcolname_woopt/scores_val.csv'

TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

The `CHECKPOINT_EPOCH` is to evaluate the weight model at a specific checkpoint. To evaluate the final weight, set is to `None`

The above configuation for evaluate evaluation set. To evaluate test set
* swith `SET_NAME` from `'val_paths'` to `'test_paths'`
* switch `FINETUNE_PATH` from `'rs_tvaev2_wcolname_woopt/finetune_val_1e-4'` to `'rs_tvaev2_wcolname_woopt/finetune_test_1e-4'`
* switch `SINGLETRAIN_PATH` from `'rs_tvaev2_wcolname_woopt/single_val_1e-4'` to `'rs_tvaev2_wcolname_woopt/single_test_1e-4'`
* switch `SCORE_SAVE_PATH` from `'rs_tvaev2_wcolname_woopt/scores_val.csv'` to `'rs_tvaev2_wcolname_woopt/scores_test.csv'`

# Run notebook to observe the report
The report contains:
* the training plot per dataset comparing between finetuning and singletraining (val set and test set)
* show the quality scores of synthetic data (as dataframe) between finetuning and singletraining (val set and test set)  

Clone the notebook `report_template.ipynb`.

Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH` to generate the training plot

Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the scores
