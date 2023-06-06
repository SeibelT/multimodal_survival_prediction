import wandb
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from utils import survival_loss
import os
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy
from torch import nn 


def MM_Trainer(model,device,epochs,trainloader,testloader,bs,lr,alpha,fold,storepath):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=[0.9,0.999],weight_decay=1e-5,)

    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-lr{lr}-fold{fold}'
    accuracy = Accuracy(task="multiclass", num_classes=4)

    with wandb.init(project='Multimodal', name=run_name, entity='tobias-seibel') as run:
        
        # Log some info
        run.config.learning_rate = lr
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model)
    
        for epoch in range(epochs):
            model.train()
            
            for idx,(histo,gen,c,l) in enumerate(trainloader):
                histo.to(device)
                gen.to(device)
                c.to(device)
                l.to(device)
                out = model(histo,gen)
                
                loss = survival_loss(out,c,l,alpha)  # TODO add loss regularization 
                loss.backward() 
                optimizer.step()
                
                with torch.no_grad():
                    #risk
                    h = nn.Sigmoid()(out)
                    S = torch.cumprod(1-h,dim = -1)
                    risk = 1-S.sum(dim=1)
                    c_index = concordance_index_censored(1-c,l,risk).item()
                    run.log({'loss': loss.item(),"accuracy": accuracy(out,l).item(),"c-index" : c_index},commit=False)
                    if idx%20==0:  # TODO still hardcoded
                        run.log()

                del h,S,risk,c_index
                    
            model.eval()
            with torch.no_grad():
                for  idx,(histo,gen,c,l) in enumerate(testloader):
                    histo.to(device)
                    gen.to(device)
                    c.to(device)
                    l.to(device)
                    out = model(histo,gen)
                    
                    loss = survival_loss(out,c,l,alpha)

                    
                    h = nn.Sigmoid()(out)
                    S = torch.cumprod(1-h,dim = -1)
                    risk = 1-S.sum(dim=1)
                    c_index = concordance_index_censored(1-c,l,risk).item()
                    run.log({'val_loss': loss.item(),"val_accuracy": accuracy(out,l).item(),"val_c-index" : c_index},commit=True)
            
            
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