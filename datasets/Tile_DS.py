import os 
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils.Aggregation_Utils import *
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl



class TileModule(pl.LightningDataModule):
    def __init__(self,df_path_train,df_path_test,df_path_val,tile_df_path,batch_size,num_workers,pin_memory,histonly=False,add_i1k=False,**kwargs):
        super().__init__()
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.df_path_val = df_path_val
        self.tile_df_path=tile_df_path
        self.num_workers = num_workers
        self.batch_size = batch_size 
        self.pin_memory = pin_memory
        self.histonly = histonly
        self.add_i1k = add_i1k
        self.transform_train =  transforms.Compose([transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5)
                                                    ]
                                                   )
        self.transform_eval =  transforms.Compose([transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    ]
                                                   )
        
    def setup(self, stage):
        if self.histonly:
            self.train_set = Tile_only_joined_Dataset(df_path=self.df_path_train,tile_df_path=self.tile_df_path,trainmode = "train",transform=self.transform_train,add_i1k=self.add_i1k)
            self.test_set = Tile_only_joined_Dataset(df_path=self.df_path_test,tile_df_path=self.tile_df_path,trainmode = "test",transform=self.transform_eval,add_i1k=False)
            self.val_set = Tile_only_joined_Dataset(df_path=self.df_path_val,tile_df_path=self.tile_df_path,trainmode = "val",transform=self.transform_eval,add_i1k=False)
            
        else:
            self.train_set = TileDataset(df_path=self.df_path_train,tile_df_path=self.tile_df_path,trainmode = "train",transform=self.transform_train)
            self.test_set = TileDataset(df_path=self.df_path_test,tile_df_path=self.tile_df_path,trainmode = "test",transform=self.transform_eval)
            self.val_set = TileDataset(df_path=self.df_path_val,tile_df_path=self.tile_df_path,trainmode = "val",transform=self.transform_eval)
            
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.pin_memory)
    
    def predict_dataloader(self):
        return ... # TODO




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
        diction= dict([(name,idx) for idx,name in enumerate(self.df_meta["slide_id"]) ]) #since slide_id is unique 
        df_tiles.insert(2,"slideid_idx",df_tiles["slide_id"].map(diction))
        df_tiles = df_tiles.dropna()
        df_tiles.slideid_idx = df_tiles.slideid_idx.astype(int)
        self.df_tiles = df_tiles
        
        
        self.transforms = transform
    def __len__(self):
        return len(self.df_tiles)
    def __getitem__(self,idx):
        
        tile_path,slide_idx = self.df_tiles.iat[idx,0],self.df_tiles.iat[idx,2]
        tile = Image.open(tile_path)
        tile = self.transforms(tile)
        
        label = torch.tensor(self.df_meta.iat[slide_idx, 1]).type(torch.int64)
        censorship = torch.tensor(self.df_meta.iat[slide_idx, 2]).type(torch.int64)
        label_cont = torch.tensor(self.df_meta.iat[slide_idx,3]).type(torch.float32)
        return (tile, self.genomics_tensor[slide_idx], censorship, label,label_cont)
        
class Patient_Tileset(Dataset):
    def __init__(self,df_tiles_path,gen_vec,transform):
        self.df_tiles_path = df_tiles_path        
        self.gen_vec = gen_vec 
        
        self.transform = transform
    def __len__(self):
        return len(self.df_tiles_path)
    def __getitem__(self,idx):
        path = self.df_tiles_path.iat[idx]
        tile = self.transform(Image.open(path))
        return (tile,self.gen_vec)
        
        
    
class Tile_only_joined_Dataset(Dataset):
    def __init__(self,df_path,tile_df_path,trainmode,transform,add_i1k):
        """Custom Dataset for Feature Extractor Finetuning for Survival Analysis 

        Args:
            df_path (str): Path to Dataframe which contains meta data and genomic data 
            tilepath (str): path to folder which contains subfolders with tiles(subfolder names must ne slide id)
            ext (str): file extension of tiles(eg jpg or png)
            trainmode (Bool): To generate train set or test set 
        """
        super(Tile_only_joined_Dataset,self).__init__()
        #Genomic Tensor and Meta Dataframe
        df = pd.read_csv(df_path) 
        
        assert trainmode in ["train","test","val"], "Dataset mode not known"
        df[df["traintest"]==(0 if trainmode=="train" else 1 if trainmode=="test" else 2)]
        
            
        
        df_meta = df[["slide_id","survival_months_discretized","censorship","survival_months"]]  
        
        # Tile Data Frame
        df_tiles = pd.read_csv(tile_df_path)
        
        # add slide_id to index mapping
        diction= dict([(name,idx) for idx,name in enumerate(df_meta["slide_id"]) ]) #since slide_id is unique 
        df_tiles.insert(2,"slideid_idx",df_tiles["slide_id"].map(diction))
        df_tiles = df_tiles.dropna()
        df_tiles.slideid_idx = df_tiles.slideid_idx.astype(int)
        
        self.df_tiles = df_tiles
        #add imagenewt1k
        if add_i1k:
            df_I1k_paths = "/globalwork/datasets/ILSVRC2015/Data/DET/train/ILSVRC2013_train" # around 300k I1k
            paths = []
            for root, dirs, files in os.walk(df_I1k_paths, topdown=False):
                for name in files:
                    if name.endswith("JPEG"):
                        paths.append(os.path.join(root, name))
            df_I1k = pd.DataFrame({"I1kpaths":paths})
            self.df_tiles = pd.concat((self.df_tiles.tilepath,df_I1k.I1kpaths))
            self.df_tiles = self.df_tiles.sample(frac=1,random_state=1337)
        else: 
            self.df_tiles = df_tiles.tilepath.sample(frac=1,random_state=1337)	
        self.transforms = transform
        
    def __len__(self):
        return len(self.df_tiles)
    
    def __getitem__(self,idx):
        tile_path = self.df_tiles.iat[idx]
        tile = Image.open(tile_path).convert('RGB')
        tile = self.transforms(tile)
        return (tile,0,0,0,0)
    
