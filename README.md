

<div align="center">
<br/>
<p align="center">
    <img src="logo.jpeg" width=400>
</p>

<h1> TabularFM: An Open Framework For Tabular Foundational Models </h1>

<span><a href="https://tabularfm.github.io" target="_blank">Official webite & Leaderboards</a></span>
</div>

TabularFM is an end-to-end framework for Tabular Foundational Models. We provide functions to train, finetune, evaluate on large amount of tabular datasets and tools to visualize and analyze foundational models. A wide range of learning methods are also supported. We have released leaderboards for Tabular Foundation Models, [visit our website](https://tabularfm.github.io) for more details. 

# Update: We have released our pretrained tabular foundational models!
The following models are relased for Gittables datasets on Huggingface:
* STVAE ([Download](https://huggingface.co/lamthuy/stvae_gittables))
* CTGAN ([Download](https://huggingface.co/lamthuy/ctgan_gittables))
* GReaT ([Download](https://huggingface.co/lamthuy/great_gittables))


# Getting Started

## Install the environment

```python
conda create -n venv python=3.9
conda activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

## Download the datasets
We have cleaned, processed and released two comprehensive datasets: [Kaggle](https://drive.google.com/drive/folders/1HnRTMBbX9kTUiDZ4pjNSWaM5SJLUSULx?usp=drive_link) and [Gittables](https://drive.google.com/file/d/10jBLjilKI5MJ_qXyDKxJFfN9ez9y9ydv/view?usp=drive_link)

* Create directory to store the datasets  
`mkdir datasets`  
`mkdir datasets/kaggle`  *Kaggle datasets*  
`mkdir datasets/gittables` *Gittables datasets*

* Download the corresponding datasets  

*If you want to use other datasets, you should clean, transform and generate metadata to be compatible with our framework.*

***We will release the code to automatically process datasets soon. Stay stuned!***

# Usage

*So far, we support learning methods: CTGAN, TVAE, STVAE, STVAEM, GReaT*

## Command-line interfrace
We provide an end-to-end CLI to run experiments  

`python -m tabularfm -mt <model_type> -d <path to datasets directory> -s <path to result directory> -c <path to config file>`  

* **-mt**: model type, we currently support `ctgan`, `tvae`, `stvae`, `stvaem`, `great`
* **-d**: path to the directory datasets, note that this directory store sub-directories of corresponding datasets. The datasets should be priorly processed and transformed. We have already provided the processed datasets of Kaggle and Gittables.
* **-s**: path to store the result of the experiment. This directory will consitsts of sub-directories corresponding to `pretrain`, `finetune`, `fromscratch`, `evaluation`
* **-c**: path to configuration file (`yaml` format) of corresponding model type (-mt). This file consists of configuration to run the whole process of the experiment. We provided sample configurations for supported methods in `configs/`

If you want to run specific training process(es), use the following additonal flags:
* **--pretrain**: pretraining
* **--finetune**: finetuning
* **--fromscratch**: training from scratch
* **--evaluate**: evaluation

### Example

The following command-line will run the experiment of STVAE  

`python -m tabularfm -mt "stvae" -d "datasets/kaggle/" -s "results_stvae" -c "stvae.yaml"`

The configuration file `stvae.yaml` is

```yaml
split_path: 'split_3sets.json' # if None, auto split the datasets
split_set: # 'val', 'test', or leave empty to run both
split_random_state: 121 # if split_path is None, split data following this random state
verbose: True
model_cfg:
  embedding_dim: 128
  encoder_dims: [512, 256, 256, 128]
  decoder_dims: [128, 256, 256, 512]
pretrain_cfg:
  epochs: 10
  batch_size: 500
  lr: 1.e-4
  optimizers: 'adam'
  checkpoint_n_epoch: 20 # checkpoint every n epochs
finetune_cfg:
  epochs: 10
  batch_size: 500
  lr: 1.e-4
  optimizers: 'adam'
  early_stopping: True # early stop in fine-tuning and single-training
fromscratch_cfg:
  epochs: 10
  batch_size: 500
  lr: 1.e-4
  optimizers: 'adam'
  early_stopping: True # early stop in fine-tuning and single-training
```

### CLI all configuration

```bash
usage: __main__.py [-h] [--pretrain | --no-pretrain] [--finetune | --no-finetune] [--fromscratch | --no-fromscratch] [--evaluate | --no-evaluate] [-mt MODEL]
                   [-d DATA_PATH] [-s SAVE_PATH] [-c CONFIG] [--resume | --no-resume]

TabularFM Command Line Interface

optional arguments:
  -h, --help            show this help message and exit
  --pretrain, --no-pretrain
                        Run pretraining
  --finetune, --no-finetune
                        Run finetuning
  --fromscratch, --no-fromscratch
                        Run training from scratch
  --evaluate, --no-evaluate
                        Run evaluation
  -mt MODEL, --model MODEL
                        Path to the training data
  -d DATA_PATH, --data DATA_PATH
                        Path to the directory of datasets
  -s SAVE_PATH, --save SAVE_PATH
                        Save directory
  -c CONFIG, --config CONFIG
                        Path to the configuration file
  --resume, --no-resume
                        Whether to resume training or not
```

## Modules
TBU

