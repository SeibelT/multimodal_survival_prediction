import torch 
from torch import nn 
from sksurv.metrics import concordance_index_censored
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
import os
import pandas as pd
from torchvision import transforms
import h5py
from datasets.Tile_DS import Patient_Tileset


def c_index(logits_all,c_all,l_all):
    """
    Variables
    logits_all : FloatTensor must be of shape = (N,4)  predicted logits of model 
    c_all : IntTensor must be of shape = (N,) 
    l_all IntTensor must be of shape = (N,)

    Outputs the c-index score 
    """
    
        
    h = nn.Sigmoid()(torch.cat(logits_all,dim=0))
    S = torch.cumprod(1-h,dim = -1)
    risk = -S.sum(dim=1) 
    notc = (1-torch.cat(c_all,dim=0)).cpu().numpy().astype(bool)
    try:
        output = concordance_index_censored(notc, torch.cat(l_all,dim=0).cpu(),risk.cpu())
        c_inidex = output[0]
    except:
        print("WARNING: C-INDEX ISSUE ,probably all samples are censored, return NAN")
        c_index = float('nan')
    return c_index

class Survival_Loss(nn.Module):
    def __init__(self,alpha,ffcv,eps = 1e-7):
        super(Survival_Loss, self).__init__()
        self.alpha = alpha
        self.eps= eps
        self.ffcv = ffcv
    def forward(self,out,c,t):
        """
        'Bias in Cross-Entropy-Based Training of Deep Survival Networks' by S.Zadeh and M.Schmid 
        https://pubmed.ncbi.nlm.nih.gov/32149626/
        Improved negative loglikeliehood loss  
        
        Variables:
        out : torch.FloatTensor  output logits of the model 
        c : torch.BoolTensor wether the patient is censored(c=1) or ucensored(c=0)
        t : torch.IntTensor label/ground truth of the index where the time-to-event is nested
        alpha : float value within [0,1] weighting the Loss of the censored patients 
        """
        assert out.device == c.device
        h = nn.Sigmoid()(out) #   Hazard function 
        S = torch.cumprod(1-h,dim = -1)  # Survival function
        
        if not self.ffcv:
            t = t.unsqueeze(-1)
        S_bar = torch.cat((torch.ones_like(t,device=t.device),S),dim=-1) # padded survival function to get acess to the previous time window 

        # gathering the probabilty within the bin with the ground truth index for hazard,survival and padded survival 
        S = S.gather(dim=-1,index=t).clamp(min=self.eps)
        h = h.gather(dim=-1,index=t).clamp(min=self.eps)
        S_bar = S_bar.gather(dim=-1,index = t).clamp(min=self.eps)
        
        
        
        #applying log function  
        logS = torch.log(S)
        logS_bar = torch.log(S_bar)
        logh = torch.log(h).to(c.device)

        #masking by censored or uncensored 
        # L_z(h,S) -> h,S_bar only uncensored,
        # while  L_censored(S) only censored 
        
        logh = logh*(1-c)
        logS_bar = logS_bar*(1-c)
        logS = logS*c

        
        L_z = -torch.mean(logh+logS_bar) # -torch.sum(logS_bar) # only uncensored needed! for h and S_bar 
        L_censored = -torch.mean(logS) # only censored S 
        
        
        return L_z + (1-self.alpha)*L_censored

def create_feature_ds(save_path,new_ds_name,model,transform,df_tile_slide_path,df_data_path,gen,batch_size,num_workers,pin_memory,cntd=False):
    """
    Args:
        save_path (str): Path where the new dataset will be stored
        new_ds_name (str): Name of new datasetfolder
        model (pl.LightningModule): 
        df_tile_slide_path (str): path to df, which contains tile and slide info 
        df_data_path (str): path to dataframe containing meta- and genomic-data 
        gen (bool): add genomic information to encoding 
        ctd (bool): If set to True, will continue on existing dataset
        
    """    
    assert os.path.exists(save_path),"Save path doesnt exist"
    save_path = os.path.join(save_path,new_ds_name)
    if cntd:
        assert os.path.exists(save_path), "Dataset does nott exist, to continue"
    else:
        assert not os.path.exists(save_path), "Dataset already exists\n Give new name or choose ctd=False to continue."
        os.mkdir(save_path)
    print("save_path :",save_path,"\ndf_tile_slide_path : ",df_tile_slide_path,"\ndf_data_path : ",df_data_path,"\n gen : ",gen,"\n ctd : ",cntd)
    #load datadframes
    df_tile_paths = pd.read_csv(df_tile_slide_path) 
    df_trainset = pd.read_csv(df_data_path)
    
    #init trainer
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
           )
    
    #add coords to all tile paths df
    df_tile_paths["coords"] = df_tile_paths.tilepath.apply(lambda x: list(map(int,x.split("(")[-1].split(")")[0].split(","))))
    #create genomics tensor 
    genomics_tensor = torch.Tensor(df_trainset[df_trainset.keys()[11:]].to_numpy()).to(torch.float32)

    for idx,slide_name in enumerate(df_trainset["slide_id"]):
        
        save_path_i = os.path.join(save_path,slide_name.replace(".svs",".h5"))
        if os.path.exists(save_path_i):
            print(f"Skip slide {slide_name},bag already exists")
            continue
        
        print(f"Init encoding for: {slide_name} ")
        #slide_name = df_trainset["slide_id"].iat[idx]
        df_tiles = df_tile_paths[df_tile_paths["slide_id"]==slide_name]
        
        if len(df_tiles)==0:
            print(f"No tiles found for:\n {slide_name}")
            continue
        
        coords_tensor = torch.tensor(list(df_tiles["coords"])).to(torch.int64)
        gen_vec = genomics_tensor[idx] if gen else torch.rand(size=(0,192))
        #dataloader for ith patient
        dataload_i = DataLoader(Patient_Tileset(df_tiles["tilepath"],gen_vec,transform), batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
        #encode features
        ##predictions = trainer.predict(model,dataload_i)
        ##feats = torch.cat(predictions,dim=0)
        
        outputs = []
        model.eval()
        for idx_eval,batch  in enumerate(dataload_i):
            x,y = batch 
            with torch.no_grad():
                outputs.append(model.predict_step(batch,idx_eval))
        feats = torch.cat(outputs)
        
        
        
        assert not feats.isnan().any().item(),f"ERROR! NaN Encoding detected at {idx} in {slide_name}"
        torch.cuda.empty_cache()
        save_path_i = os.path.join(save_path,slide_name.replace(".svs",".h5"))
        with h5py.File(save_path_i, "w") as f:
                    coords = f.create_dataset("coords", data=coords_tensor.to(torch.int64))
                    feats = f.create_dataset("feats", data=feats.to(torch.float16))
        
