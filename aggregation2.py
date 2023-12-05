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
    storepath = config["storepath"]
    num_workers = config["num_workers"]
    do_test = config["do_test"] if "do_test" in config else True
    do_val = config["do_val"] if "do_val" in config else True
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
        if do_val:
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
        if do_val:
            val_ds = HistGen_Dataset(df_val,data_path = feature_path,train=False,mode="val"if kfold is None else "kfold")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = Porpoise(d_hist=dim_hist,d_gen=d_gen,d_gen_out=32,device=device,activation=activation,bins=bins,d_hidden=d_hidden).to(device)
        
    
    elif modality=="PrePorpoise":
        train_ds = HistGen_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        if do_val:
            val_ds = HistGen_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = PrePorpoise(d_hist=dim_hist,d_gen=d_gen,d_transformer=512,dropout=dropout,activation=activation,bins=bins,d_hidden=d_hidden).to(device)
        
    elif modality=="PrePorpoise_meanagg_attmil":
        train_ds = HistGen_Dataset(df_train,data_path = feature_path,mode="train")
        if do_val:
            val_ds = HistGen_Dataset(df_val,data_path = feature_path,mode="val")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = PrePorpoise_meanagg(d_hist=dim_hist,d_gen=d_gen,d_transformer=512,dropout=dropout,activation=activation,bins=bins,d_hidden=d_hidden,attmil=True).to(device)
    
    elif modality=="PrePorpoise_meanagg":
        train_ds = HistGen_Dataset(df_train,data_path = feature_path,mode="train")
        if do_val:
            val_ds = HistGen_Dataset(df_val,data_path = feature_path,mode="val")
        if do_test:
            test_ds = HistGen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = PrePorpoise_meanagg(d_hist=dim_hist,d_gen=d_gen,d_transformer=512,dropout=dropout,activation=activation,bins=bins,d_hidden=d_hidden,attmil=False).to(device)
    
    elif modality=="gen":
        train_ds = Gen_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        if do_val:
            val_ds = Gen_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = Gen_Dataset(df_test,data_path = feature_path,mode="test")
        d_gen = train_ds.gen_depth()
        model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device,activation=activation,d_hidden=d_hidden).to(device)
        
    
    elif modality=="hist":
        
        train_ds = Hist_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        if do_val:
            val_ds = Hist_Dataset(df_val,data_path = feature_path,train=False,mode="val" if kfold is None else "kfold")
        if do_test:
            test_ds = Hist_Dataset(df_test,data_path = feature_path,mode="test")
        model = AttMil_Survival(d_hist=dim_hist,bins=bins,device=device,d_hidden=d_hidden).to(device)
        
    elif modality=="hist_attention":
        train_ds = Hist_Dataset(df_train,data_path = feature_path,train=True,mode="train" if kfold is None else "kfold")
        if do_val:
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
    if do_val:
        val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=True)
    else: 
        val_dataloader = None 
        
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
    
    #run.watch(model,log_freq=1,log="all")
    #run trainer
    if modality in ["Porpoise","PrePorpoise","PrePorpoise_meanagg_attmil","PrePorpoise_meanagg"]:
        _,risk_all,c_all,l_con_all = MM_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,testloader=test_dataloader
                    )
        
    elif modality in ["gen","hist","hist_attention"]:
        
        _,risk_all,c_all,l_con_all = Uni_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,testloader=test_dataloader
                    )
    
    #do table 
    #tablepath = os.path.join(storepath,f'fold{config["csv_path"].split("/")[-1]}')
    #os.makedirs(tablepath, exist_ok = True) 
    #tablepath = os.path.join(tablepath,f"{modality}_lr{learningrate}_{feature_path.split('/')[-2]}.csv")
    df = pd.DataFrame({"risk":risk_all,"c":c_all,"l_cont":l_con_all})
    df["fold"] = config["csv_path"].split("/")[-1]
    df["modality"] =  modality
    df["learningrate"] = learningrate
    #df.to_csv(tablepath)
    run.log({"risk_score_table":wandb.Table(dataframe=df)})
            
if __name__ == "__main__":    
    
    aggregation()