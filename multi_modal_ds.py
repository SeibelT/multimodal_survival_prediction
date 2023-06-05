from torch.utils.data import Dataset
import pandas as pd 
import torch 
import h5py
import os 

class HistGen_Dataset(Dataset):
    def __init__(self,csv_path,data_path,train):
        # no transformation needed 
        self.df = pd.csv_read(csv_path)
        self.data_path = data_path
        
        if train: 
            self.df = self.df[self.df['kfold']>0]
            # 
        else: 
            self.df = self.df[self.df['kfold']==0]

        self.df["filepath"]

        
        self.genomics_tensor = torch.Tensor(self.df[self.df.keys()[11:]].asarray)
        
        self.df = self.df[["path","survival_months_discretized","censorship"]]


    def __len__(self):
        return len(self.df)


    def __getitem__(self,idx):
        
        tensor_path =os.path.join(self.data_path, self.df.iloc[idx, 0]+".h5") 
        tensor_file =h5py.File(tensor_path, "r")
        label = torch.tensor(self.df.iloc[idx, 1])
        censorship = torch.tensor(self.df.iloc[idx, 2]).astype(torch.int16)

        return tensor_file, self.genomics_tensor[idx], censorship,  label

        
