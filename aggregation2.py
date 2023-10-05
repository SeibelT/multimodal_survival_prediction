import os
import wandb
import torch
import collections
import pandas as pd
import multiprocessing
import argparse
import yaml
from models.Aggregation_Models import *
from trainer.Aggregation_Trainer import *
from datasets.Aggregation_DS import *
from pytorch_lightning.loggers import WandbLogger
from utils.Aggregation_Utils import Survival_Loss
            
def dropmissing(df,name,feature_path):
        len_df = len(df)
        df = df.drop(df[df["slide_id"].apply(lambda x : not os.path.exists(os.path.join(feature_path,x.replace(".svs",".h5"))))].index)
        if len_df !=len(df):
            print(f"Dropped {len_df-len(df)}rows in {name} dataframe")
        return df
    
def aggregation():
    run = wandb.init()
    
    config=dict(run.config)
    run_name = run.name or  "unknown"
    
    batchsize = config["batchsize"]
    bins = config["bins"]
    d_gen_out = config["d_gen_out"]
    epochs = config["epochs"] 
    learningrate = config["learningrate"]
    alpha = config["alpha"]
    l1_lambda = config["l1_lambda"]
    activation = config["activation"]
    modality = config["modality"]
    dropout = config["dropout"]
    d_hidden = config["d_hidden"]
    dim_hist,feature_path = config["dim_hist_and_feature_path"] 
    storepath = os.path.join(config["storepath"],f"{modality}sweep")  
    num_workers = config["num_workers"]
    do_test = True
    kfold = config["kfold"] 
    num_fold = 5
    if kfold is not None:
        csv_path_kfold = os.path.join(config["csv_path"],"tcga_brca_trainable"+str(bins)+".csv") # CSV file 
        df_kfold = pd.read_csv(csv_path_kfold)
        df_kfold = dropmissing(df_kfold,"kfold",feature_path)
        df_kfold["kfold"] = df_kfold["kfold"].apply(lambda x : (x+kfold)%num_fold) 
        df_kfold = df_kfold.sample(frac=1,random_state=1337)
        df_train = df_kfold
        df_val = df_kfold
        do_test = False
        
        #assert not do_test, "kfold not applicable with test"
    else:
        csv_path_train = os.path.join(config["csv_path"],f"tcga_brca__{bins}bins_trainsplit.csv") 
        df_train = pd.read_csv(csv_path_train)
        df_train = dropmissing(df_train,"train",feature_path)
        csv_path_val = os.path.join(config["csv_path"],f"tcga_brca__{bins}bins_valsplit.csv") 
        df_val = pd.read_csv(csv_path_val)
        df_val = dropmissing(df_val,"val",feature_path)
        
        if do_test:
            csv_path_test = os.path.join(config["csv_path"],f"tcga_brca__{bins}bins_testsplit.csv") 
            df_test = pd.read_csv(csv_path_test)
            df_test = dropmissing(df_test,"test",feature_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    #Initialize Dataset and Model based on Modality
    if modality=="Porpoise":
        train_ds = HistGen_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        val_ds = HistGen_Dataset(df_val,data_path = feature_path,train=False,mode="val"if kfold is None else "kfold")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = Porpoise(d_hist=dim_hist,d_gen=d_gen,d_gen_out=32,device=device,activation=activation,bins=bins,d_hidden=d_hidden).to(device)
        
    
    elif modality=="PrePorpoise":
        train_ds = HistGen_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        val_ds = HistGen_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = PrePorpoise(d_hist=dim_hist,d_gen=d_gen,d_transformer=512,dropout=dropout,activation=activation,bins=bins,d_hidden=d_hidden).to(device)
        
    
    elif modality=="gen":
        train_ds = Gen_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        val_ds = Gen_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = Gen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device,activation=activation,d_hidden=d_hidden).to(device)
        
    
    elif modality=="hist":
        train_ds = Hist_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        val_ds = Hist_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = Hist_Dataset(df_test,data_path = feature_path,mode="test")
        model = AttMil_Survival(d_hist=dim_hist,bins=bins,device=device,d_hidden=d_hidden).to(device)
        
    elif modality=="hist_attention":
        train_ds = Hist_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        val_ds = Hist_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = Hist_Dataset(df_test,data_path = feature_path,mode="test")
        model = TransformerMil_Survival(d_hist=dim_hist,bins=bins,dropout=dropout,d_hidden=d_hidden).to(device)
    
   
    criterion = Survival_Loss(alpha) 
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=True)
    if do_test:
        test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=True)
    else: 
        test_dataloader = None 
    val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
    
    #run.watch(model,log_freq=1,log="all")
    #run trainer
    if modality in ["Porpoise","PrePorpoise",]:
        c_vals = MM_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,testloader=test_dataloader
                    )
        
    elif modality in ["gen","hist","hist_attention"]:
        c_vals = Uni_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,testloader=test_dataloader
                    )
   
if __name__ == "__main__":     
    aggregation()