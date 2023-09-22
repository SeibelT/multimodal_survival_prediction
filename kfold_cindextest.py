from trainer.Aggregation_Trainer import Uni_Trainer_sweep,eval_func
from utils.Aggregation_Utils import Survival_Loss
from datasets.Aggregation_DS import Gen_Dataset
from models.Aggregation_Models import SNN_Survival
import torch
import pandas as pd
import numpy as np 

num_workers=1
batchsize = 32
epochs = 40 
l1_lambda = 1e-7
modality="gen"
alpha = 0.25
learningrate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
d_gen_out = 32
bins = 4

print(f"num_workers={num_workers}\nbatchsize = {batchsize}\nepochs = {epochs} \nl1_lambda = {l1_lambda}\nmodality={modality}\nalpha = {alpha }\nlearningrate = {learningrate}\nd_gen_out = {d_gen_out}\nbins = {bins}\n")

f = "/nodes/bevog/work4/seibel/data/aggregation_kfold_dataframes/tcga_brca_trainable4.csv"
data_path = None
df = pd.read_csv(f)

criterion = Survival_Loss(alpha) 


outputvals = []
modelweightslist = []
for j in range(5):
    outputvals_k = []
    modelweightslist_k = []
    
    df.kfold = df.kfold.apply(lambda x: (x+1)%5)
    train_ds  = Gen_Dataset(df,data_path,train=True,mode="kfold")
    val_ds  = Gen_Dataset(df,data_path,train=False,mode="kfold")
    d_gen = train_ds.gen_depth()
    trainloader = torch.utils.data.DataLoader(train_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=False)
    valloader = torch.utils.data.DataLoader(val_ds,batch_size=batchsize,num_workers=num_workers,pin_memory=False)

    for i in range(20):
        print("fold",j,"run",i)
        model = SNN_Survival(d_gen,d_gen_out,bins,device,activation="SELU").to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
        
        c_train_rand,_,_ = eval_func(model,trainloader,criterion,device,bins,"unimodal")
        c_val_rand,_,_ = eval_func(model,valloader,criterion,device,bins,"unimodal")
        
        c_train,c_val,modelweights = Uni_Trainer_sweep(run=None,model=model,optimizer=optimizer,criterion=criterion,trainloader=trainloader,
                        valloader=valloader,bins=bins,epochs=epochs,device=device,storepath=None,run_name=None,
                        l1_lambda=l1_lambda,modality=modality,testloader=None)

        outputvals_k.append([c_train,c_val,c_train_rand,c_val_rand])
        #modelweightslist_k.append(modelweights)
    outputvals.append(outputvals_k)
    #modelweightslist.append(modelweightslist_k)

print(outputvals)
print(np.mean(outputvals,axis=0))