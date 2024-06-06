from tabularfm.ctgan.synthesizers.tvae import CustomTVAE
from tabularfm.ctgan.data_transformer import DataTransformer
import pandas as pd
import pickle
import json
import os

from sklearn.model_selection import train_test_split
import numpy as np

from utils import get_df, dump_transformer

############# CONFIG #############
DATA_PATH = 'data/processed_dataset' # directory storing datasets
############# END CONFIG #############

for path in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, path)
    dump_transformer(path)