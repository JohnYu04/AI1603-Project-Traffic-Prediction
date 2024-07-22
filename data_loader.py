import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def extract_208x208_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    if df.shape != (209, 209):
        raise ValueError("CSV文件中的数据不是209x209矩阵")
    matrix_208x208 = df.iloc[1:, 1:].to_numpy()
    matrix_208x208_tensor = torch.tensor(matrix_208x208, dtype=torch.float32)
    return matrix_208x208_tensor
class Dataset_Paris(Dataset):
    def __init__(self, root_path, data_path='train.csv', seq_len=24, pred_len=1):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[cols]
        
        data = df_raw.values.transpose(0, 1)
        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end].transpose(1, 0)
        seq_y = self.data_y[r_begin:r_end].transpose(1, 0)

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class Dataset_Paris_Test(Dataset):
    def __init__(self, root_path, data_path='test_history.csv', seq_len=24, pred_len=1):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[cols]
        
        #data = df_raw.values.transpose(0, 1)
        data = df_raw.values
        data = data.reshape(-1, self.seq_len, data.shape[1])
        data = data.transpose(0, 2, 1)
        self.data_x = data

    def __getitem__(self, index):
        return self.data_x[index]

    def __len__(self):
        return len(self.data_x)

def get_dataloader(batch_size=16, root_path='.', 
                   data_path='train.csv',
                   seq_len=24,
                   pred_len=1,
                   shuffle=True,
                   num_workers=1,
                   drop_last=True):
    if data_path != 'test_history.csv':
        data_set = Dataset_Paris(
            root_path=root_path,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len)
        print('Loading from: %s' % (data_path), len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last)
    else:
        data_set = Dataset_Paris_Test(
            root_path=root_path,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len)
        print('Loading from: %s' % (data_path), len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

    return data_loader
    