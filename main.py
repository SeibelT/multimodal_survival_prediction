import sys
from utils import  prepare_csv
from models import *
from multi_modal_ds import HistGen_Dataset
import pandas as pd
import torch
from trainer import *
from torch.utils.tensorboard import SummaryWriter

def MM_train_func():
    modality="multimodal"
    f = "/nodes/bevog/work4/seibel/data/tcga_brca_trainable.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files" # folderpath for h5y fiels which contain the WSI feat vecs 
    batchsize = 1  # due to different size of bags 
    epochs = 20  # 20 in paper 
    df = pd.read_csv(f)
    learningrate = 2e-4
    alpha = 0.1 # TODO find real value!!!
    folds = 5
    storing_path = "/nodes/bevog/work4/seibel/results/alpha"+str(alpha)+modality  # Where to store the model/ TODO checkpoints
    train_ds = HistGen_Dataset(df,data_path = data_path,train=True)
    test_ds = HistGen_Dataset(df,data_path = data_path,train=False)
    d_hist = 2048
    d_gen = train_ds.gen_depth()
    d_gen_out = 32
    model = Porpoise(d_hist,d_gen,d_gen_out,device=device)  # Get settings 

    #writer = SummaryWriter(("./tensorboard"))
    #input1 = torch.randn(1,10,d_hist)
    #input2 = torch.randn(1,d_gen)
    #input1 = input1.to(device).float()
    #input2 = input2.to(device).float()
    #model.to(device)
    #writer.add_graph(model, (input1,input2))
    
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
    
    MM_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
               testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,
               fold= folds,storepath=storing_path,l1_lambda = 0.0005)
    

        

    
def MM_train_func_kfold():
    modality = "gen" # must be one of ["multimodal","hist","gen"]
    
    f = "/nodes/bevog/work4/seibel/data/tcga_brca_trainable.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files" # folderpath for h5y fiels which contain the WSI feat vecs 
    batchsize = 1  # due to different size of bags 
    epochs = 20  # 20 in paper 
    df = pd.read_csv(f)
    learningrate = 2e-4
    alpha = 0.3 # TODO find real value!!!
    folds = 5
    storing_path = '/work4/seibel/multimodal_survival_prediction/results/'+modality  # Where to store the model/ TODO checkpoints

    for fold in range(folds):  
        df["kfold"] = df["kfold"].apply(lambda x : (x+1)%folds)

        train_ds = HistGen_Dataset(df,data_path = data_path,train=True)
        test_ds = HistGen_Dataset(df,data_path = data_path,train=False)

        d_hist = 2048
        d_gen = train_ds.gen_depth()
        d_gen_out = 32

        
        training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
        test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
        
        
        if modality =="multimodal":
            model = Porpoise(d_hist,d_gen,d_gen_out,device=device)
            MM_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,modality=modality)
        elif modality =="gen":
            model = SNN_Survival(d_gen,d_gen_out,device=device)
            Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,modality=modality)
        elif modality =="hist":
            model = AttMil_Survival(d_hist,device=device)
            Unimodal_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
                testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,
                fold= fold,storepath=storing_path,modality=modality)
        


def prep_CSV(f):
    """
    give path to csv zip file to get a trainable csv 
    """
    savename = f.replace("all_clean.csv.zip","trainable.csv")
    prepare_csv(f,k=5,n_bins=4,savename = savename)
    print("Finished")

if __name__ == '__main__':
    
  
  #print(sys.argv)
  #functions = {'train': MM_train_func,'trainkfold': MM_train_func_kfold,'prepcsv':prep_CSV}
  #functions[sys.argv[1]](sys.argv[2])
  
  #MM_train_func()
  MM_train_func_kfold()
  
