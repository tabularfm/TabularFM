# Tabsyn v3

# Changelog
- 24.04: Update code for training CustomTVAE, OriginalTVAE, CTGAN, TVAE with column name embeddings, transform v3

# Datasets
- [x] [Kaggle](https://drive.google.com/drive/folders/1HnRTMBbX9kTUiDZ4pjNSWaM5SJLUSULx?usp=drive_link)
- [x] 1M Gittables Datasets: [ref](https://gittables.github.io/), [processed data](https://drive.google.com/file/d/1JcmHi1lacXgcjjyGo9-JS3ZBGnu2szO3/view?usp=drive_link)

# Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r latest_requirements.txt
```

# Download datasets
Create a data directory: `mkdir data`  
Download and store the dataset, e.g. `data/processed_dataset/`

# Preprocess datasets

## Clean and generate metadata  

If the data was not cleaned before, clean the data  
`python preprocess_data.py`

Configuration

```python
SOURCE_DIR = 'data/processed_dataset' # data source
TARGET_DIR = 'data/processed_dataset' # target directory to save processed data

PREPROCESS_CONFIG = {
    'clean_data': True,
    'clean_exclude_columns': [], # exclude chosen columns to apply cleaning, if empty, all columns are considered
    'min_freq_threshold': 0.03, # if max frequency of a values of a column >= this param, all values of the column will be kept
    'percentage_to_remove': 0.9, # if the number of categories in the column >= this param * number of data rows, it will be removed
    'percentage_to_remove_row': 0.5,
    'generate_metadata': True,
}
```

## Transform data  
`python transform_data.py`

Configuration

```python
DATA_PATH = 'data/processed_dataset' # directory storing datasets
```

## Split data 
Split data into 3 sets: pretraining, validation, and test sets  
`python split_datasets.py`

Configuration

```python
DATA_PATH = 'data/processed_dataset' # data source
SAVE_PATH = 'split_3sets.json' # save path to store splitting info
PRETRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
```

Splitting info will be stored in `SAVE_PATH` as a json structure with 3 keys: `pretrain_paths`, `val_paths`, `test_paths`. Each key contains a list of dataset names.

# Pretraining
Train a model on a large dataset (pretraining)  
`python pretrain_v2.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset' # source dataset
SAVE_PATH = 'rs_v2/pretraining_1e-4' # directory to save
SPLIT_INFO_PATH = 'split_3sets.json' # split info

TOTAL_EPOCHS = 500 # total training epoch
CHECKPOINT_EPOCH = 10 # save after every checkpoint epoch
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)

```

The result will be stored in the `SAVE_PATH`, the directory contains
* `encoder_checkpoint_<n_epoch>.pt` and `decoder_checkpoint_<n_epoch>.pt` for every `CHECKPOINT_EPOCH`
* `encoder_weights.pt` and `decoder_weights.pt`: final weights after completing the training
* `training_hist.csv`: training and validation loss of the training

# Finetuning
Finetune with a pretrained model per dataset (finetuning). The training procedure already applied early stopping.  
`python finetune_v2.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
PRETRAIN_PATH = 'rs/pretraining_1e-4' # directory that store a pretrained model
SAVE_PATH = 'rs/finetune_test_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths

TOTAL_EPOCHS = 5
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

The result will be stored in the `SAVE_PATH`, the directory contains
* `<dataset_name>_encoder.pt` and `<dataset_name>_decoder.pt` are weights of finetuned model corresponding to the dataset.
* `training_hist.csv`: training and validation loss of the training

# Single training per dataset
Train a model per dataset (single training)  
`python singletrain_v2.py`

Configuration
```python
DATA_PATH= 'data/processed_dataset'
SAVE_PATH = 'rs/single_test_1e-4'
SPLIT_INFO_PATH = 'split_3sets.json'
SET_NAME = 'val_paths' # val_paths / test_paths 

TOTAL_EPOCHS = 5
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

# Evaluate the synthetic data
Generate data and compare the quality of synthetic data of finetuning vs. singletraining. The training procedure already applied early stopping.  
`python evaluate_syndata_v2.py`

Configuration
```python
DATA_PATH = 'data/processed_dataset' # directory of source (real) data
SPLIT_INFO_PATH = 'split_3sets.json' # split info path
SET_NAME = 'test_paths' # val_paths or test_paths
FINETUNE_PATH = 'rs_v2/finetune_test_1e-4' # directory to save the scores of finetuning (as csv)
SINGLETRAIN_PATH = 'rs_v2/single_test_1e-4' # directoryto save the scores of singletraining (as csv)
SCORE_SAVE_PATH = 'rs_v2/scores_test.csv' # path to save the score (concatenated of the above scores) of finetuning and singletraining

# Model config
TOTAL_EPOCHS = 500
BATCH_SIZE = 500
LR = 1.e-4
EMBEDDING_DIM = 128
ENCODERS_DIMS = (512, 256, 256, 128)
DECODER_DIMS = (128, 256, 256, 512)
```

# Run notebook to observe the report
Access the notebook `report_template.ipynb` to generate
* the training plot per dataset comparing between finetuning and singletraining (val set and test set)
* show the quality scores of synthetic data (as dataframe) between finetuning and singletraining (val set and test set)

Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH` to generate the training plot

Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres
