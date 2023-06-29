import sys
from utils import  prepare_csv
from models import *
from multi_modal_ds import HistGen_Dataset
import pandas as pd
import torch
from trainer import *
from torch.utils.tensorboard import SummaryWriter

def MM_train_func(modality, epochs,learningrate,alpha,folds, d_gen_out,l1_lambda,d_hist,bins=4,transformer=False):
    
    
    #Dont Change:
    f = f"/nodes/bevog/work4/seibel/data/tcga_brca_trainable{bins}.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files" # folderpath for h5y fiels which contain the WSI feat vecs 
    batchsize = 1  # due to different size of bags 
    df = pd.read_csv(f)
    storing_path = "/nodes/bevog/work4/seibel/results/alpha"+str(alpha)+modality  # Where to store the model/ TODO checkpoints
    train_ds = HistGen_Dataset(df,data_path = data_path,train=True)
    test_ds = HistGen_Dataset(df,data_path = data_path,train=False)
    
    d_gen = train_ds.gen_depth()
    
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
        

    if modality =="multimodal":
            model = Porpoise(d_hist,d_gen,d_gen_out,bins=bins,device=device)
            MM_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,lr=learningrate,alpha=alpha ,
                fold= folds,storepath=storing_path,l1_lambda= l1_lambda,bins=bins)
    elif modality =="gen":
        model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device)
        Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
            testloader=test_dataloader,lr=learningrate,alpha=alpha ,
            fold= folds,storepath=storing_path,modality=modality,l1_lambda = l1_lambda,bins=bins)
    elif modality =="hist":
        if transformer:
            model = TransformerMil_Survival(d_hist,bins=bins,device=device)
        else:
            model = AttMil_Survival(d_hist,bins=bins,device=device)
        Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
            testloader=test_dataloader,lr=learningrate,alpha=alpha ,
            fold= folds,storepath=storing_path,modality=modality,l1_lambda = l1_lambda,bins=bins)
        

    
def MM_train_func_kfold(modality, epochs,learningrate,alpha,folds, d_gen_out,l1_lambda,d_hist,bins,transformer=False):
    
    #Dont change 
    f = f"/nodes/bevog/work4/seibel/data/tcga_brca_trainable{bins}.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files" # folderpath for h5y fiels which contain the WSI feat vecs 
    batchsize = 1  # due to different size of bags 
    
    df = pd.read_csv(f)
    storing_path = '/work4/seibel/multimodal_survival_prediction/results/'+modality  # Where to store the model/ TODO checkpoints

    for fold in range(folds):  
        df["kfold"] = df["kfold"].apply(lambda x : (x+1)%folds)

        train_ds = HistGen_Dataset(df,data_path = data_path,train=True)
        test_ds = HistGen_Dataset(df,data_path = data_path,train=False)

        
        d_gen = train_ds.gen_depth()
        test_ds.gen_depth()

        
        training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
        test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
        
        
        if modality =="multimodal":
            model = Porpoise(d_hist,d_gen,d_gen_out,bins=bins,device=device)
            MM_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,l1_lambda= l1_lambda,bins=bins)
        elif modality =="gen":
            model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device)
            Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,modality=modality,l1_lambda = l1_lambda,bins=bins)
        elif modality =="hist":
            if transformer:
                model = TransformerMil_Survival(d_hist,bins=bins,device=device)
            else:
                model = AttMil_Survival(d_hist,bins=bins,device=device)
            Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,modality=modality,l1_lambda = l1_lambda,bins=bins)
        


def prep_CSV(f,bins=4):
    """
    give path to csv zip file to get a trainable csv 
    """
    savename = f.replace("all_clean.csv.zip",f"trainable{bins}.csv")
    prepare_csv(f,k=5,n_bins=bins,savename = savename)
    print("Finished")

if __name__ == '__main__':


    #print(sys.argv)
    #functions = {'train': MM_train_func,'trainkfold': MM_train_func_kfold,'prepcsv':prep_CSV}
    #functions[sys.argv[1]](sys.argv[2])

    #prep_CSV(f ="/work4/seibel/PORPOISE/datasets_csv/tcga_brca_all_clean.csv.zip" ,bins=64)
    #prep_CSV(f ="/work4/seibel/PORPOISE/datasets_csv/tcga_brca_all_clean.csv.zip" ,bins=32)
    #prep_CSV(f ="/work4/seibel/PORPOISE/datasets_csv/tcga_brca_all_clean.csv.zip" ,bins=16)
    #prep_CSV(f ="/work4/seibel/PORPOISE/datasets_csv/tcga_brca_all_clean.csv.zip" ,bins=8)
    


    modality = "hist" # must be one of ["multimodal","hist","gen"]
    epochs = 20  # 20 in paper 
    learningrate = 2e-4   
    
    folds = 5
    d_gen_out = 32
    bins=4
    d_hist = 2048
    alpha = 0.5 
    l1_lambda =1e-7
    
    #parameter search mesh 
    for bins in [8,16,32]:
        for learningrate in [2e-4,2e-5]:
            
            MM_train_func(modality, epochs,learningrate,alpha,folds, d_gen_out,l1_lambda,d_hist,bins=bins )
            
    ##best params kfolds 
    #for alpha in [0.5,1]:
    #    for l1_lambda in [1e-5]:
    #        MM_train_func_kfold(modality, epochs,learningrate,alpha,folds, d_gen_out,l1_lambda,d_hist,bins=4)