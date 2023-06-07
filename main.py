import sys
from utils import  prepare_csv
from models import Porpoise
from multi_modal_ds import HistGen_Dataset
import pandas as pd
import torch
from trainer import MM_Trainer


def MM_train_func():
    f = "/nodes/bevog/work4/seibel/data/tcga_brca_trainable.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/xiyue-wang" # folderpath for h5y fiels which contain the WSI feat vecs 
    batchsize = 1  # due to different size of bags 
    epochs = 10  # 20 in paper 
    df = pd.read_csv(f)
    learningrate = 2e-4
    alpha = 0.3 # TODO find real value!!!
    folds = 5
    storing_path = "/nodes/bevog/work4/seibel/results/test1"  # Where to store the model/ TODO checkpoints
    train_ds = HistGen_Dataset(df,data_path = data_path,train=True)
    test_ds = HistGen_Dataset(df,data_path = data_path,train=False)
    d_hist = 2048
    d_gen = train_ds.gen_depth()
    d_gen_out = 32
    model = Porpoise(d_hist,d_gen,d_gen_out,device=device)  # Get settings 
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
    
    MM_Trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,
               testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,
               fold= folds,storepath=storing_path)
    
    """
    folds = folds 
    for fold in range(folds):  
        df["kfold"] = df["kfold"].apply(lambda x : (x+1)%folds)

        train_ds = HistGen_Dataset(df,datapath = data_path,train=True)
        test_ds = HistGen_Dataset(df,datapath = data_path,train=False)

        model = Porpoise()
        training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
        test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
        
        trainer(model=model,device=device,epochs= epochs,trainloader=training_dataloader,testloader=test_dataloader,bs=batchsize,lr=learningrate,alpha=alpha ,fold= folds)
    """
        

    



def prep_CSV(f):
    """
    give path to csv zip file to get a trainable csv 
    """
    savename = f.replace("all_clean.csv.zip","trainable.csv")
    prepare_csv(f,k=5,n_bins=4,savename = savename)
    print("Finished")

if __name__ == '__main__':
    
  
  #print(sys.argv)
  #functions = {'mmtrain': MM_train_func,'prepcsv':prep_CSV}
  #functions[sys.argv[1]](sys.argv[2])
  
  MM_train_func()
  
