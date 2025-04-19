import os
import numpy as np
import pandas as pd
import glob
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.utils import shuffle
from utils.tools import print_and_save
from itertools import combinations
from collections import defaultdict


import warnings
warnings.filterwarnings('ignore')


class Dataset_Text_Earnings(Dataset):
    def __init__(self, root_path, data_path, flag,
                 emb_vendor, emb_file, y_column, rolling_test):
        """
        Initialize the dataset with configuration parameters.
        
        Args:
            root_path (str): Base directory for all data
            data_path (str): Path to the earnings CSV file, relative to root_path
            flag (str): Dataset split - must be 'train', 'test', or 'val'
            emb_vendor (str): Provider of embeddings (e.g., 'openai')
            emb_file (str): Base filename for the embeddings (without extension)
            y_column (str): Column name for the target volatility values
            rolling_test (str): Identifier for the rolling test configuration
        """
        valid_flags = ['train', 'test', 'val']
        assert flag in valid_flags, f"Flag must be one of {valid_flags}, got '{flag}'"

        self.root_path = root_path
        self.data_path = data_path
        self.emb_vendor = emb_vendor
        self.emb_file = emb_file
        self.y_column = y_column
        self.rolling_test = rolling_test
        self.cate_column = f'rolling_test_on_{rolling_test}_cate'
        self.flag = flag
        
        # Initialize tracking variables
        self.ids = None
        self.verbose = True
        
        # Load and process data
        self.__read_data__()

    def __read_data__(self):
        """
        Read the earnings data and corresponding embeddings.
        
        This method loads the earnings CSV, identifies the ID column,
        extracts target values, and loads the appropriate embeddings.
        """
        # Load the main earnings dataframe
        self.df_earnings = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Determine and validate ID column ("id" in DEC and "text_file_name" in EC and MAEC)
        self.id_column = 'id' if 'id' in self.df_earnings.columns else 'text_file_name'
        assert self.id_column in self.df_earnings.columns, f"Required ID column '{self.id_column}' missing from dataframe"
        
        # Extract and validate target values
        assert self.y_column in self.df_earnings.columns, f"Target column '{self.y_column}' missing from dataframe"
        self.volatility = self.df_earnings[self.y_column].values
        
        # Handle datasets that use 'dev' (in EC and MAEC) instead of 'val' (in DEC) for evaluation
        if self.flag == 'val' and 'val' not in self.df_earnings[self.cate_column].unique():
            self.flag = 'dev'
            if self.verbose:
                print(f"Note: Changing flag from 'val' to 'dev' based on available categories")

        # Load embeddings and filter the data
        self.text_embeddings = self._load_embeddings()
        self._filter_data()

    def _load_embeddings(self):
        """
        Load text embeddings from the specified vendor and file.
        
        Returns:
            np.ndarray: Array of text embeddings
        """
        # Construct path to embedding file
        emb_path = os.path.join(self.root_path, "Embeddings", self.emb_vendor, f"{self.emb_file}.npz")
        
        # Load embeddings and IDs
        with np.load(emb_path, allow_pickle=True) as data:
            emb, ids = data['embeddings'], data['ids']
        
        # Verify alignment between embeddings and dataframe
        assert sum(ids == self.df_earnings[self.id_column]) == len(self.df_earnings), \
            "Mismatch between embedding IDs and dataframe IDs"
        
        return emb

    def _filter_data(self):
        """
        Filter and assign embeddings and target values based on the current flag (train/val/test).
        
        This method subsets the data to include only the relevant split based on the cate_column.
        """
        # Remove entries marked as 'none' (after testing earnings)
        self.df_earnings = self.df_earnings[self.df_earnings[self.cate_column] != 'none']
        
        # Filter to only include entries for the current flag
        cate_df = self.df_earnings[self.df_earnings[self.cate_column] == self.flag]
        
        # Log information if verbose
        if self.verbose:            
            print(f'\nRolling test on: {self.rolling_test}, '
                  f'{len(cate_df)}/{len(self.df_earnings)} earnings are valid for category: {self.flag}')

        # Extract embeddings and targets for the current category
        self.cate_text_embeddings = self.text_embeddings[cate_df.index]
        self.cate_volatility = self.volatility[cate_df.index]

        # For test set, store IDs for reporting
        if self.flag == 'test':
            self.ids = cate_df[self.id_column].values

    def __getitem__(self, index):
        """
        Get a single item from the dataset by index.
        
        Args:
            index (int): Index of the item
            
        Returns:
            tuple: (embedding, volatility_target)
        """
        return self.cate_text_embeddings[index], self.cate_volatility[index]

    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items in the dataset
        """
        return len(self.cate_text_embeddings)




class Dataset_TimeSeires_Earnings(Dataset):
    def __init__(self, root_path, data_path, size,
                 ts_pattern, scale, rolling_test, enc_in, prediction_window,
                 flag='train', features='MS', timeenc=0, freq='b',
                 seasonal_patterns=None, is_training=1):

        # size: [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.enc_in = enc_in
        self.prediction_window = prediction_window

        assert flag in ['train', 'test', 'val']
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.rolling_test = rolling_test
        self.cate_column = f'rolling_test_on_{rolling_test}_cate'
        self.ts_pattern = ts_pattern

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.verbose = True
        self.__read_data__()

    def __read_data__(self):
        if self.verbose:
            print(f'\nInitiating Dataset_Earnings with flag: {self.flag}')
            print(f'seq_len: {self.seq_len}, label_len: {self.label_len}, pred_len: {self.pred_len}')

        self.df_earnings = pd.read_csv(os.path.join(self.root_path, self.data_path))

        self.id_column = 'id' if 'id' in self.df_earnings.columns else 'text_file_name'
        assert self.id_column in self.df_earnings.columns, "ID column missing."

        self._filter_data()

  
    def _filter_data(self):
        self.io_columns = {
            'volatility': {
                'input_columns': {
                    'volatility_past': [f'lv{self.prediction_window}_past_{i}' for i in range(self.seq_len, 0, -1)],
                },
                'output_columns': {
                    'volatility_future': [f'lv{self.prediction_window}_future_{self.prediction_window}'],
                }
            },
        }
        assert self.ts_pattern in self.io_columns.keys()
        input_columns = self.io_columns[self.ts_pattern]['input_columns']
        output_columns = self.io_columns[self.ts_pattern]['output_columns']

        self.df_earnings = self.df_earnings[self.df_earnings[self.cate_column] != 'none']
        cate_df = self.df_earnings[self.df_earnings[self.cate_column] == self.flag]

        ts_data = [cate_df[input_columns[c]] for c in input_columns]
        ts_data = np.stack(ts_data, axis=-1)

        volatility_data = [cate_df[output_columns[c]] for c in output_columns]                
        volatility_data = np.stack(volatility_data, axis=-1)
        
        self.timeseries = np.concatenate((ts_data, volatility_data), axis=1)

        if self.verbose:            
            print(f'\nRolling test on: {self.rolling_test}, {len(cate_df)}/{len(self.df_earnings)} earnings are valid for cate: {self.flag}!')
            print(f'X.shape: {ts_data.shape}')
            print(f'Y.shape: {volatility_data.shape}')
        
        if self.flag == 'test':
            self.ids = cate_df[self.id_column]

    def __getitem__(self, index):
        # Masks are data stamps
        X = self.timeseries[index][:self.seq_len].reshape(-1, self.enc_in)
        y = self.timeseries[index][self.seq_len - self.label_len:
                                   self.seq_len + self.pred_len].reshape(-1, self.enc_in)

        X_mask = np.zeros_like(X)
        y_mask = np.zeros_like(X)
        return X, y, X_mask, y_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
