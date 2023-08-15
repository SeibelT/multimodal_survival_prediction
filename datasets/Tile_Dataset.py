import os 
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils.Aggregation_Utils import *
from torch.utils.data import DataLoader,Dataset

import pytorch_lightning as pl



class TileModule(pl.LightningDataModule):
    def __init__(self,df_path_train,df_path_test,df_path_val,tile_df_path,batch_size=32):
        super().__init__()
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.df_path_val = df_path_val
        self.tile_df_path=tile_df_path
        
        self.batch_size = batch_size 
        self.transform_train =  transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5)
                                                    ]
                                                   )
        self.transform_eval =  transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    ]
                                                   )
        
    def setup(self, stage):
        self.train_set = TileDataset(df_path=self.df_path_train,tile_df_path=self.tile_df_path,trainmode = "train",transform=self.transform_train)
        self.test_set = TileDataset(df_path=self.df_path_test,tile_df_path=self.tile_df_path,trainmode = "test",transform=self.transform_eval)
        self.val_set = TileDataset(df_path=self.df_path_val,tile_df_path=self.tile_df_path,trainmode = "val",transform=self.transform_eval)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,num_workers=6,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,num_workers=6,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,num_workers=6,pin_memory=True)




class TileDataset(Dataset):
    def __init__(self,df_path,tile_df_path,trainmode,transform):
        """Custom Dataset for Feature Extractor Finetuning for Survival Analysis 

        Args:
            df_path (str): Path to Dataframe which contains meta data and genomic data 
            tilepath (str): path to folder which contains subfolders with tiles(subfolder names must ne slide id)
            ext (str): file extension of tiles(eg jpg or png)
            trainmode (Bool): To generate train set or test set 
        """
        super(TileDataset,self).__init__()
        #Genomic Tensor and Meta Dataframe
        df = pd.read_csv(df_path) 

        assert trainmode in ["train","test","val"], "Dataset mode not known"
        df[df["traintest"]==(0 if trainmode=="train" else 1 if trainmode=="test" else 2)]
        
            
        self.genomics_tensor = torch.Tensor(df[df.keys()[11:]].to_numpy()).to(torch.float32) # 11 is hardcoded for this cohort, might differ for other cohorts.
        self.df_meta = df[["slide_id","survival_months_discretized","censorship","survival_months"]]  
        
        # Tile Data Frame
        df_tiles = pd.read_csv(tile_df_path)
        
        # add slide_id to index mapping
        diction= dict([(name,idx) for idx,name in enumerate(self.df_meta["slide_id"]) ]) 
        df_tiles.insert(2,"slideid_idx",df_tiles["slide_id"].map(diction))
        df_tiles = df_tiles.dropna()
        df_tiles.slideid_idx = df_tiles.slideid_idx.astype(int)
        self.df_tiles = df_tiles
        
        # TODO transforms 
        self.transforms = transform
    def __len__(self):
        return len(self.df_tiles)
    def __getitem__(self,idx):
        
        tile_path,_,slide_idx = self.df_tiles.iloc[idx]
        tile = Image.open(tile_path)
        tile = self.transforms(tile)
        
        label = torch.tensor(self.df_meta.iloc[slide_idx, 1]).type(torch.int64)
        censorship = torch.tensor(self.df_meta.iloc[slide_idx, 2]).type(torch.int64)
        label_cont = torch.tensor(self.df_meta.iloc[slide_idx,3]).type(torch.int64)
        return tile, self.genomics_tensor[slide_idx], censorship, label,label_cont
        