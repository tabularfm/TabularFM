

<div align="center">
<br/>
<p align="center">
    <img src="logo.jpeg" width=400>
</p>

<h1> TabularFM: An Open Framework For Tabular Foundational Models </h1>

<span><a href="https://tabularfm.github.io" target="_blank">Official webite & Leaderboards</a></span>
</div>

# Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

# Usage

We support learning methods: `ctgan`, `tvae`, `stvae`, `stvaem`, `great`

#### Pretraining
```python
python tabularfm/pretrain.py -m <model_type> -d <datasets directory> -s <save directory> -c <pretraining configuration>
```

Example:
```python
python tabularfm/pretrain.py -m "stvae" -d "datasets/kaggle/" -s "pretrain_stvae/" -c "tabularfm/configs/pt_stvae.yaml"
```


#### Finetuning
```python
python tabularfm/finetune.py -m <model_type> -d <datasets directory> -p <pretrained model directory>  -s <save directory> -c <finetuning configuration>
```

Example:
```python
python tabularfm/finetune.py -m "stvae" -d "datasets/kaggle/" -p "pretrain_stvae/" -s "finetune_stvae/" -c "tabularfm/configs/ft_stvae.yaml"
```

#### Single training
*TBU*

# Supported Datasets
* [Kaggle](https://drive.google.com/drive/folders/1HnRTMBbX9kTUiDZ4pjNSWaM5SJLUSULx?usp=drive_link)
* [1M Gittables Datasets](https://drive.google.com/file/d/10jBLjilKI5MJ_qXyDKxJFfN9ez9y9ydv/view?usp=drive_link)


<!-- # Note
* Set up directories before run the experiment
    * Create directory to store the result for each methods: `mkdir rs_<method_name>_<optional_info>/`
    * Inside the created directory, create directories for pretraining, finetune (val and test), singletrain (val and test)
    
* Change `SPLIT_INFO_PATH` to change the split information
    * For Kaggle datasets: `split_3sets.json`
    * For Gittables datasets: `split_3sets_gittables.json`
    
* Change `DATA_PATH` to change dataset directory
* In pretraining, if the training is interrupted, set `RESUME_TRAINING` to True before re-run the script


# Original TVAE
## Pretraining
* `python pretrain_oritvae`

## Finetuning
* `python finetune_oritvae.py`

## Single training
* `python singletrain_oritvae.py`

## Evaluate
* `python evaluate_syndata_oritvae.py`

## Report
* Clone `report_template.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

# CustomTVAE (STVAE)
## Pretraining
* `python pretrain_v2`

## Finetuning
* `python finetune_v2.py`

## Single training
* `python singletrain_v2.py`

## Evaluate
* `python evaluate_syndata_v2.py`

## Report
* Clone `report_template.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

# CustomTVAE with colname emebdding WITHOUT optimization (STVAE (M))
## Pretraining
* `python pretrain_tvae_wcolname_woopt.py`

## Finetuning
* `python finetune_tvae_wcolname_woopt.py`

## Single training
* `python singletrain_tvae_wcolname_woopt.py`

## Evaluate
* `python evaluate_syndata_tvae_wcolname_woopt.py`

## Report
* Clone `report_template.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

# CustomTVAE with colname emebdding WITH optimization (STVAE (MO))
## Pretraining
* `python pretrain_tvae_wcolname.py`

## Finetuning
* `python finetune_tvae_wcolname.py`

## Single training
* `python singletrain_tvae_wcolname.py`

## Evaluate
* `python evaluate_syndata_tvae_wcolname.py`

## Report
* Clone `report_template.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

# CTGAN
## Pretraining
* `python pretrain_ctgan.py`

## Finetuning
* `python finetune_ctgan.py`

## Single training
* `python singletrain_ctgan.py`

## Evaluate
* `python evaluate_syndata_ctgan.py`

## Report
* Clone `report_template_gan.ipynb` and set name
* Replace `FINETUNE_PATH` and `SINGLETRAIN_PATH`
* Replace `VAL_SCORE_PATH` and `TEST_SCORE_PATH` to show the socres

# GReaT (TBU) -->