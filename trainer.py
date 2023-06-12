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



def MM_Trainer(model,device,epochs,trainloader,testloader,bs,lr,alpha,fold,storepath,l1_lambda=None):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=[0.9,0.999],weight_decay=1e-5,)
    criterion = Survival_Loss(alpha)
    #criterion = nn.CrossEntropyLoss() #CE loss 
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}_{"papersettings"}_l1_{l1_lambda}_both"-fold{fold}-lr{lr}' # TODO still hard coded 
    accuracy = Accuracy(task="multiclass", num_classes=4,average='weighted')
    
    #tensorboard
    
    

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
            
            #init counter
            out_all =torch.zeros(size=(len(trainloader),4),device='cpu')      
            c_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            l_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            runningloss = 0

            for idx,(histo,gen,c,l) in enumerate(trainloader):
                
                histo = histo.to(device)
                gen = gen.to(device)
                optimizer.zero_grad()

                out = model(histo,gen)
                out = out.cpu()
                
                if l1_lambda is None:# TODO add to settings on lambda 
                    """no l1 reg"""
                    
                    #loss = criterion(out,l)  ## for CE loss 
                    loss = criterion(out,c,l)  # TODO add loss regularization 
                else:
                    """l1 reg"""
                    histo_weights = torch.cat([x.flatten() for x in model.Attn_Mil.parameters()])
                    gen_weights = torch.cat([x.flatten() for x in model.SNN.parameters()])
                    #loss = criterion(out,l)  ## for CE loss 
                    loss = criterion(out,c,l) + l1_lambda * torch.norm(histo_weights,1) + l1_lambda * torch.norm(gen_weights,1)
                         
                
                loss.backward() 
                optimizer.step()
                runningloss += loss.item()
                
                #add to counter
                out_all[idx,:] = out
                l_all[idx] = l
                c_all[idx] = c

                run.log({"step":epoch*len(trainloader)+idx+1})
                del out,l,c

            c_index_train = c_index(out_all,c_all,l_all)


            #init counter
            out_all_val =torch.empty(size=(len(testloader),4),device='cpu')        
            l_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            c_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            val_rloss = 0

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
                    l_all_val[idx] = l
                    c_all_val[idx] = c


            c_index_val = c_index(out_all_val,c_all_val,l_all_val)

            wandbdict = {"epoch": epoch+1,'train/runningloss': runningloss/len(testloader),
                         "train/accuracy":accuracy(out_all,l_all).item(),
                         "train/auc": roc_auc_score(l_all.detach().numpy(),nn.Softmax(dim=-1)(out_all).detach().numpy(),average="macro",multi_class="ovo"),
                         "train/c_index":c_index_train,
                         'valid/runningloss': val_rloss/len(testloader),
                         "valid/accuracy":accuracy(out_all_val,l_all_val).item(),
                         "valid/auc": roc_auc_score(l_all_val.detach().numpy(),nn.Softmax(dim=-1)(out_all_val).detach().numpy(),average="macro",multi_class="ovo"),
                         "valid/c_index":c_index_val,
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

def c_index(out_all,c_all,l_all): # TODO to utils 
    """
    Variables
    out_all : FloatTensor must be of shape = (N,4)  predicted logits of model 
    c_all : IntTensor must be of shape = (N,) 
    l_all IntTensor must be of shape = (N,)

    Outputs the c-index score 
    """
    with torch.no_grad():  # TODO c-index not working yet 
                    #risk
                    h = nn.Sigmoid()(out_all)
                    S = torch.cumprod(1-h,dim = -1)
                    risk = -S.sum(dim=1) ## TODO why is it not 1-S ???
                    notc = (1-c_all).numpy().astype(bool)
                    try:
                        c_index = concordance_index_censored(notc,l_all.cpu(),risk)
                        #print(c_index)
                    except:
                        print("C index problems")
                    return c_index[0]