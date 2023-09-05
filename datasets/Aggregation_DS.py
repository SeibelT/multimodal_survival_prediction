from torch.utils.data import Dataset
import pandas as pd 
import torch 
import h5py
import os 

class HistGen_Dataset(Dataset):
    def __init__(self,df,data_path,train=None,mode="kfold"):
        # no transformation needed 
        self.df = df 
        self.data_path = data_path
        if mode == "kfold":
            if train: 
                self.df = self.df[self.df['kfold']>0]
                # 
            else: 
                self.df = self.df[self.df['kfold']==0]
        
        elif mode == "train":
            self.df = self.df[self.df['traintest']==0]
        elif mode == "test":
            self.df = self.df[self.df['traintest']==1]
        elif mode == "val":
            self.df = self.df[self.df['traintest']==2]
    
        self.genomics_tensor = torch.Tensor(self.df[self.df.keys()[11:]].to_numpy()).to(torch.float32)
        
        self.df = self.df[["slide_id","survival_months_discretized","censorship","survival_months"]]


    def __len__(self):
        return len(self.df)

    def gen_depth(self):
        _,depth = self.genomics_tensor.size()
        return depth 
    
     
    def __getitem__(self,idx):
        
        tensor_path =os.path.join(self.data_path, self.df.iat[idx, 0].replace(".svs",".h5")) 
        tensor_file =h5py.File(tensor_path, "r")
        tensor_file = torch.tensor(tensor_file["feats"][:]).to(torch.float32)
        label = torch.tensor(self.df.iat[idx, 1]).type(torch.int64)
        censorship = torch.tensor(self.df.iat[idx, 2]).type(torch.int64)
        label_cont = torch.tensor(self.df.iat[idx,3]).type(torch.float32)
        return tensor_file, self.genomics_tensor[idx], censorship,  label,label_cont

        

class Gen_Dataset(Dataset):
    def __init__(self,df,data_path,train=None,mode="kfold"):
        # no transformation needed 
        self.df = df 
        self.data_path = data_path
        if mode == "kfold":
            if train: 
                self.df = self.df[self.df['kfold']>0]
                # 
            else: 
                self.df = self.df[self.df['kfold']==0]

        elif mode == "train":
            self.df = self.df[self.df['traintest']==0]
        elif mode == "test":
            self.df = self.df[self.df['traintest']==1]
        elif mode == "val":
            self.df = self.df[self.df['traintest']==2]
            
        self.genomics_tensor = torch.Tensor(self.df[self.df.keys()[11:]].to_numpy()).to(torch.float32)
        self.df = self.df[["slide_id","survival_months_discretized","censorship","survival_months"]]


    def __len__(self):
        return len(self.df)

    def gen_depth(self):
        _,depth = self.genomics_tensor.size()
        return depth 
    
     
    def __getitem__(self,idx):
        
        label = torch.tensor(self.df.iat[idx, 1]).type(torch.int64)
        censorship = torch.tensor(self.df.iat[idx, 2]).type(torch.int64)
        label_cont = torch.tensor(self.df.iat[idx,3]).type(torch.float32)
        return self.genomics_tensor[idx], censorship,  label,label_cont

        

class Hist_Dataset(Dataset):
    def __init__(self,df,data_path,train=None,mode="kfold"):
        # no transformation needed 
        self.df = df 
        self.data_path = data_path
        
        if mode == "kfold":
            if train: 
                self.df = self.df[self.df['kfold']>0]
                # 
            else: 
                self.df = self.df[self.df['kfold']==0]
        
        elif mode == "train":
            self.df = self.df[self.df['traintest']==0]
        elif mode == "test":
            self.df = self.df[self.df['traintest']==1]
        elif mode == "val":
            self.df = self.df[self.df['traintest']==2]
            
        self.df = self.df[["slide_id","survival_months_discretized","censorship","survival_months"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        tensor_path =os.path.join(self.data_path, self.df.iat[idx, 0].replace(".svs",".h5")) 
        tensor_file =h5py.File(tensor_path, "r")
        tensor_file = torch.tensor(tensor_file["feats"][:]).to(torch.float32)
        
        label = torch.tensor(self.df.iat[idx, 1]).type(torch.int64)
        censorship = torch.tensor(self.df.iat[idx, 2]).type(torch.int64)
        label_cont = torch.tensor(self.df.iat[idx,3]).type(torch.float32)
        return tensor_file, censorship,  label,label_cont

        