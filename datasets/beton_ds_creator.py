#copied and adapted from https://github.com/SerezD/ffcv_pytorch_lightning
#
import os 
from ffcv.fields import RGBImageField,NDArrayField,FloatField
from ffcv_pl.generate_dataset import create_beton_wrapper
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import torch
import pandas as pd

class TileDataset(Dataset):
    def __init__(self,df_path,tile_df_path,trainmode):
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
        
            
        self.genomics_tensor = np.ascontiguousarray(df[df.keys()[11:]],dtype=np.dtype(np.float32))#.to_numpy(dtype=np.dtype(np.float32)) # 11 is hardcoded for this cohort, might differ for other cohorts.
        self.df_meta = df[["slide_id","survival_months_discretized","censorship","survival_months"]]  
        
        # Tile Data Frame
        df_tiles = pd.read_csv(tile_df_path)
        
        # add slide_id to index mapping
        diction= dict([(name,idx) for idx,name in enumerate(self.df_meta["slide_id"]) ]) 
        df_tiles.insert(2,"slideid_idx",df_tiles["slide_id"].map(diction))
        df_tiles = df_tiles.dropna()
        df_tiles.slideid_idx = df_tiles.slideid_idx.astype(int)
        self.df_tiles = df_tiles
        
    def __len__(self):
        return len(self.df_tiles)
    def __getitem__(self,idx):
        
        tile_path,_,slide_idx = self.df_tiles.iloc[idx]
        tile = Image.open(tile_path)
        
        
        label = int(self.df_meta.iloc[slide_idx, 1])
        censorship = int(self.df_meta.iloc[slide_idx, 2])
        label_cont = float(self.df_meta.iloc[slide_idx,3])
        return (tile, self.genomics_tensor[slide_idx], censorship, label,label_cont)



def main(trainmode = "val"):
    
    # 1. Instantiate the torch dataset that you want to create
    # Important: the __get_item__ dataset must return tuples! (This depends on FFCV library)
    df_path_train = "/nodes/bevog/work4/seibel/PORPOISE/datasets_csv/tcga_brca__4bins_trainsplit.csv"
    df_path_test = "/nodes/bevog/work4/seibel/PORPOISE/datasets_csv/tcga_brca__4bins_testsplit.csv"
    df_path_val = "/nodes/bevog/work4/seibel/PORPOISE/datasets_csv/tcga_brca__4bins_valsplit.csv"
    tile_df_path = "/nodes/bevog/work4/seibel/multimodal_survival_prediction/datasets/DF_TCGA-BRCA-TIILES-NORM.csv"
    
    if trainmode == "test":
        safe_path = "/globalwork/seibel/beton_ds/tile_survival_test.beton"
        image_label_dataset = TileDataset(df_path_test,tile_df_path,trainmode)
    if trainmode == "train":
        safe_path = "/globalwork/seibel/beton_ds/tile_survival_train.beton"
        image_label_dataset = TileDataset(df_path_train,tile_df_path,trainmode)
    if trainmode == "val":
        safe_path = "/globalwork/seibel/beton_ds/tile_survival_val.beton"
        image_label_dataset = TileDataset(df_path_val,tile_df_path,trainmode)
    
    
    
    # 2. Optional: create Field objects.
    # here overwrites only RGBImageField, leave default IntField.
    fields = (RGBImageField(write_mode='jpg', jpeg_quality=95),  NDArrayField(shape=(20971,),dtype=np.dtype(np.float32)), None,None,FloatField())
    
    # 3. call the method, and it will automatically create the .beton dataset for you.
    if os.path.exists(safe_path):
        print("file already exists")
    else:
        print(f"Safe beton wrapper {trainmode}")
        create_beton_wrapper(image_label_dataset, safe_path, fields)


if __name__ == '__main__':
    #main(trainmode = "train") remove comment before running
    ...