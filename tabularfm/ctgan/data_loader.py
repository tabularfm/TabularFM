import torch
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset
from tabularfm.ctgan.data_transformer import DataTransformer, ColnameTransformer
import torch.nn.functional as F

import pandas as pd


class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors: Tensor, colname_embeddings: Tensor, max_dim: int, device) -> None:
        super().__init__(*tensors)
        
        self.colname_emebddings = torch.squeeze(colname_embeddings)
        self.max_dim = max_dim
        self.device = device
        
    def __getitem__(self, index):
        # stack (transfomed_data, colname_embedding, zero-padding) 
        return tuple(F.pad(torch.hstack((tensor[index], self.colname_emebddings)), 
                           (0, self.max_dim - len(tensor[index]) - len(self.colname_emebddings))).to(self.device) for tensor in self.tensors)
        
class CustomTensorDatasetV2(Dataset):
    def __init__(self, df_path, colname_embeddings: bool, max_dim: int, transformer, chunksize=1e4, df_indices=None) -> None:
        super(CustomTensorDatasetV2, self).__init__()
        
        self.df_path = df_path
        self.transformer = transformer
        self.colname_transformer = ColnameTransformer() if colname_embeddings else None
        self.max_dim = max_dim
        self.chunksize = chunksize
        self.df_indices = df_indices
        
        self.get_actual_df_len()
        
    def get_actual_df_len(self):
        len = 0
        for i, chunk in enumerate(pd.read_csv(self.df_path, chunksize=self.chunksize)):
            if self.df_indices is not None:
                chunk_indices = self.df_indices[(self.df_indices >= i * self.chunksize) & (self.df_indices < len(chunk))]
                len += len(chunk_indices)
            else:
                len += len(chunk)
                
        self.len_df = len
    
    def get_item_from_chunk(self, index):
        
        for i, chunk in enumerate(pd.read_csv(self.df_path, chunksize=self.chunksize)):
            if i * self.chunksize<= index < (i+1) * self.chunksize:
                data_row_df = chunk.iloc[[index]]
                colname_texts = data_row_df.columns.to_list()
                
                tensor_data = self.transformer.transform(data_row_df)
                tensor_data = torch.from_numpy(tensor_data)
                
                colname_embedding = self.colname_transformer.transform(colname_texts).detach().numpy().reshape(1, -1)
                colname_embedding = torch.from_numpy(colname_embedding)
                
                tensor_data = torch.hstack((tensor_data, colname_embedding))
                output = F.pad(tensor_data, (0, self.max_dim - len(tensor_data) - len(colname_embedding)))
                
                return output
            else:
                continue
            
    def __len__(self):
        return self.len_df
        
    def __getitem__(self, index):
        return tuple(self.get_item_from_chunk(index))
