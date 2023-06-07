import wandb
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from utils import Survival_Loss
import os
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy
from torch import nn 


def MM_Trainer(model,device,epochs,trainloader,testloader,bs,lr,alpha,fold,storepath):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=[0.9,0.999],weight_decay=1e-5,)
    criterion = Survival_Loss(alpha)
    #criterion = nn.CrossEntropyLoss() CE loss as test if my loss is wrong 
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-lr{lr}-fold{fold}_{7}' # TODO still hard coded 
    accuracy = Accuracy(task="multiclass", num_classes=4)
    
    print(run_name)
    with wandb.init(project='MultiModal', name=run_name, entity='tobias-seibel',mode='online') as run:
        
        # Log some info
        run.config.learning_rate = lr
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model,log = 'all',criterion = criterion,log_graph=True,log_freq=10)
        
        
        

        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("valid/*", step_metric="epoch")
        
        
        
        for epoch in range(epochs):
            model.train()
            
            out_all =torch.empty(size=(len(trainloader),4),device='cpu')  # 4 for bins still hardcoded TODO         
            c_all = torch.empty(size=(len(trainloader),1),device='cpu')
            l_all = torch.empty(size=(len(trainloader),1),device='cpu').to(torch.int16)
            runningloss = 0
            for idx,(histo,gen,c,l) in enumerate(trainloader):
                
                histo = histo.to(device)
                gen = gen.to(device)
                optimizer.zero_grad()
                out = model(histo,gen)

                out = out.cpu()
                #loss = criterion(out,l)  ## for CE loss 
                loss = criterion(out,c,l)  # TODO add loss regularization 
                loss.backward() 
                optimizer.step()
                runningloss += loss.item()
                
                out_all[idx,:] = out
                l_all[idx,:] = l
                c_all[idx,:] = c

                run.log({"step":epoch*len(trainloader)+idx+1})
            

            with torch.no_grad():  # TODO c-index not working yet 
                    #risk
                    h = nn.Sigmoid()(out_all)
                    S = torch.cumprod(1-h,dim = -1)
                    risk = 1-S.sum(dim=1)
                    notc = torch.logical_not(c_all.type(torch.BoolTensor))
                    
                    
            val_rloss = 0
            
            out_all_val =torch.empty(size=(len(testloader),4),device='cpu')  # 4 for bins still hardcoded TODO         
            l_all_val = torch.empty(size=(len(testloader),1),device='cpu').to(torch.int16)
            model.eval()
            with torch.no_grad():
                for  idx,(histo,gen,c,l) in enumerate(testloader):
                    histo = histo.to(device)
                    gen = gen.to(device)

                    out = model(histo,gen)

                    out = out.cpu()
                    #loss = criterion(out,l)  #CE loss
                    loss = criterion(out,c,l)  # TODO add loss regularization 
                    val_rloss += loss.item()
                    
                    
                    out_all_val[idx,:] = out
                    l_all_val[idx,:] = l


                    h = nn.Sigmoid()(out)
                    S = torch.cumprod(1-h,dim = -1)
                    risk = 1-S.sum(dim=1)
                    
                    
                    
            
            wandbdict = {"epoch": epoch,'train/runningloss': runningloss/len(testloader),
                         "train/accuracy":accuracy(out_all,l_all.squeeze(dim=-1)).item(),
                         'valid/runningloss': val_rloss/len(testloader),
                         "valid/accuracy":accuracy(out_all_val,l_all_val.squeeze(dim=-1)).item(),
                        }
            run.log(wandbdict)

            
            
            #if epoch%10 and epoch!=0:
            #    store_checkpoint(epoch,model,optimizer,storepath,run_name)  
    
    # Store models with fold name  
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    


def store_checkpoint(epoch,model,optimizer,storepath,run_name):
    torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                os.path.join(storepath, f"{run_name}_{epoch}.pth"))
