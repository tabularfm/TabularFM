import pandas as pd
import csv
import os
import numpy as np
import re
from pathlib import Path
import pickle
import re

def get_df(path):
    try:
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", index_col=0)
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1")
        df = pd.read_csv(path, on_bad_lines='skip', encoding = 'utf-8')
    except:
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", lineterminator='\n', quoting=csv.QUOTE_NONE, index_col=0)
        # df = pd.read_csv(path, on_bad_lines='skip', encoding = "ISO-8859-1", quoting=csv.QUOTE_NONE, index_col=0)
        df = pd.read_csv(path, on_bad_lines='skip', encoding = 'utf-8', lineterminator='\n', quoting=csv.QUOTE_NONE)
        
    return df

def iterate_save_large_df(df, iter, csv_path):
    md,hd='a',False
    
    if iter == 0:
        md,hd='w',True
        
    df.to_csv(csv_path,mode=md,header=hd,index=None)