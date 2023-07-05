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
from utils import *


def MM_Trainer(model,device,epochs,trainloader,testloader,lr,alpha,fold,storepath,bins,l1_lambda=None):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=[0.9,0.999],weight_decay=1e-5,)
    criterion = Survival_Loss(alpha)
    #criterion = nn.CrossEntropyLoss() #CE loss 
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}_nll-alpha{str(alpha)}-l1_lambda{l1_lambda}-fold{fold}-bins{bins}' # TODO still hard coded 
    accuracy = Accuracy(task="multiclass", num_classes=bins,average='weighted')

    print(run_name)
    with wandb.init(project='MultiModal', name=run_name, entity='tobias-seibel',mode='online') as run:
        
        # Log some info
        run.config.l1_lambda = l1_lambda
        run.config.alpha = alpha
        run.config.learning_rate = lr
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model,log = 'all',criterion = criterion,log_graph=True,log_freq=10)
        
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("valid/*", step_metric="epoch")
        
        for epoch in range(epochs):
            model.train()
            
            #init counter
            out_all =torch.zeros(size=(len(trainloader),bins),device='cpu')      
            c_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            l_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            runningloss = 0

            for idx,(histo,gen,c,l,_) in enumerate(trainloader):
                
                histo = histo.to(device)
                gen = gen.to(device)
                optimizer.zero_grad()

                out = model(histo,gen)
                out = out.cpu()
                
                if l1_lambda is None:# TODO add to settings on lambda 
                    """no l1 reg"""
                    
                    loss = criterion(out,l)  ## for CE loss 
                    #loss = criterion(out,c,l)  # TODO add loss regularization 
                else:
                    """l1 reg"""
                    histo_weights = torch.cat([x.flatten() for x in model.Attn_Mil.parameters()])
                    gen_weights = torch.cat([x.flatten() for x in model.SNN.parameters()])
                    #loss = criterion(out,l) + l1_lambda * torch.norm(gen_weights,1)    + l1_lambda * torch.norm(histo_weights,1) ## for CE loss 
                    loss = (1-2*l1_lambda)*criterion(out,c,l) + l1_lambda * torch.norm(gen_weights.cpu(),1).cpu()    + l1_lambda * torch.norm(histo_weights.cpu(),1).cpu()
                         
                
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
            out_all_val =torch.empty(size=(len(testloader),bins),device='cpu')        
            l_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            c_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            val_rloss = 0

            model.eval()
            with torch.no_grad():
                for  idx,(histo,gen,c,l,_) in enumerate(testloader):
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

    # Store models with fold name  
    if not os.path.exists(storepath):
        os.mkdir(storepath)
        
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


def Unimodal_Trainer(model,device,epochs,trainloader,testloader,lr,alpha,storepath,modality,fold,bins,l1_lambda=None):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=[0.9,0.999],weight_decay=1e-5,)
    criterion = Survival_Loss(alpha)
    #criterion = nn.CrossEntropyLoss() #CE loss 
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}_nll-alpha{str(alpha)}-fold{fold}-l1_lambda{l1_lambda}-bins{bins}' # TODO still hard coded 
    accuracy = Accuracy(task="multiclass", num_classes=bins,average='weighted')
    
    print(run_name)
    with wandb.init(project='MultiModal', name=run_name, entity='tobias-seibel',mode='online') as run:
        
        # Log some info
        run.config.l1_lambda = l1_lambda
        run.config.alpha = alpha
        run.config.learning_rate = lr
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model,log = 'all',criterion = criterion,log_graph=True,log_freq=10)
    
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("valid/*", step_metric="epoch")
        
        for epoch in range(epochs):
            model.train()
            
            #init counter
            out_all =torch.zeros(size=(len(trainloader),bins),device='cpu')      
            c_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            l_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
            runningloss = 0

            for idx,(histo,gen,c,l,_) in enumerate(trainloader):
                if modality =="hist":
                    x = histo.to(device)
                elif modality =="gen":
                    x = gen.to(device)
                else:
                    print("Wrong modalityname")
                    print(1/0)
                    
                optimizer.zero_grad()

                out = model(x)
                out = out.cpu()
                
                if l1_lambda is None:# TODO add to settings on lambda 
                    """no l1 reg"""
                    
                    #loss = criterion(out,l)  ## for CE loss 
                    loss = criterion(out,c,l)  # TODO add loss regularization 
                else:
                    """l1 reg"""
                    if modality =="hist":
                        weights = torch.cat([x.flatten() for x in model.Attn_Mil.parameters()])
                    elif modality =="gen":
                        weights = torch.cat([x.flatten() for x in model.SNN.parameters()])
                    #loss = criterion(out,l) + l1_lambda * torch.norm(weights,1)     ## for CE loss 
                    loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1)
                         
                
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
            out_all_val =torch.empty(size=(len(testloader),bins),device='cpu')        
            l_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            c_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
            val_rloss = 0

            model.eval()
            with torch.no_grad():
                for  idx,(histo,gen,c,l,_) in enumerate(testloader):
                    if modality =="hist":
                        x = histo.to(device)
                    elif modality =="gen":
                        x = gen.to(device)
                    out = model(x)
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
        
    # Store models with fold name  
    
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    
    
    
def Uni_Trainer_sweep(run,model,optimizer,criterion,trainloader,
                      testloader,bins,epochs,device,storepath,run_name,
                      l1_lambda,modality
                      ):
    

    
    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        model.train()
        
        #init counter
        out_all =torch.zeros(size=(len(trainloader),bins),device='cpu')      
        c_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
        l_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
        runningloss = 0
        
        for idx,(x,c,l,_) in enumerate(trainloader):
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = out.cpu()
            
            if modality =="gen":
                weights = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
            elif modality=="hist":
                weights = torch.cat([x.flatten() for x in model.AttMil.parameters()]) 
            
            loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1)
            loss.backward() 
            optimizer.step()
            
            runningloss += loss.item()
            
            #add to counters
            out_all[idx,:] = out
            l_all[idx] = l
            c_all[idx] = c
            del out,l,c

        c_index_train = c_index(out_all,c_all,l_all)

        #init counter
        out_all_val =torch.empty(size=(len(testloader),bins),device='cpu')        
        l_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        l_con_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        c_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        val_rloss = 0

        model.eval()
        with torch.no_grad():
            for  idx,(x,c,l,l_con) in enumerate(testloader):
                x = x.to(device)
                out = model(x)
                out = out.cpu()
                #loss = criterion(out,l)  #CE loss
                loss = criterion(out,c,l)  # TODO add loss regularization 
                val_rloss += loss.item()
                
                out_all_val[idx,:] = out
                l_all_val[idx] = l
                l_con_all_val[idx] = l_con
                c_all_val[idx] = c


        c_index_val = c_index(out_all_val,c_all_val,l_all_val)
        c_index_val_all[epoch] = c_index_val
        
        
        wandbdict = {"epoch": epoch+1,
                        "train/runningloss": runningloss/len(testloader),
                        "train/c_index":c_index_train,
                        'valid/runningloss': val_rloss/len(testloader),
                        "valid/c_index":c_index_val,
                    }
        run.log(wandbdict)
    
    KM_wandb(run,out_all_val,c_all_val,event_cond=l_con_all_val,n_thresholds = 4,nbins = 30)
    
    return c_index_val_all

    
    # Store models with fold name  
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    
    
    
def MM_Trainer_sweep(run,model,optimizer,criterion,trainloader,
                      testloader,bins,epochs,device,storepath,run_name,
                      l1_lambda
                      ):
    

    
    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        model.train()
        
        #init counter
        out_all =torch.zeros(size=(len(trainloader),bins),device='cpu')      
        c_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
        l_all = torch.zeros(size=(len(trainloader),),device='cpu').to(torch.int16)
        runningloss = 0
        
        for idx,(x,y,c,l,_) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x,y)
            out = out.cpu()
            
            weights1 = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
            weights2 = torch.cat([x.flatten() for x in model.AttMil.parameters()]) 
            
            loss = criterion(out,c,l) + 0.5 * l1_lambda * (torch.norm(weights1,1)+torch.norm(weights2,1)).cpu()
            loss.backward() 
            optimizer.step()
             
            runningloss += loss.item()
            
            #add to counters
            out_all[idx,:] = out
            l_all[idx] = l
            c_all[idx] = c
            del out,l,c

        c_index_train = c_index(out_all,c_all,l_all)

        #init counter
        out_all_val =torch.empty(size=(len(testloader),bins),device='cpu')        
        l_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        l_con_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        c_all_val = torch.empty(size=(len(testloader),),device='cpu').to(torch.int16)
        val_rloss = 0

        model.eval()
        with torch.no_grad():
            for  idx,(x,c,l,l_con) in enumerate(testloader):
                x = x.to(device)
                out = model(x)
                out = out.cpu()
                #loss = criterion(out,l)  #CE loss
                loss = criterion(out,c,l)  # TODO add loss regularization 
                val_rloss += loss.item()
                
                out_all_val[idx,:] = out
                l_all_val[idx] = l
                l_con_all_val[idx] = l_con
                c_all_val[idx] = c


        c_index_val = c_index(out_all_val,c_all_val,l_all_val)
        c_index_val_all[epoch] = c_index_val
        
        
        wandbdict = {"epoch": epoch+1,
                        "train/runningloss": runningloss/len(testloader),
                        "train/c_index":c_index_train,
                        'valid/runningloss': val_rloss/len(testloader),
                        "valid/c_index":c_index_val,
                    }
        run.log(wandbdict)
    
    KM_wandb(run,out_all_val,c_all_val,event_cond=l_con_all_val,n_thresholds = 4,nbins = 30)
    
    return c_index_val_all

    
    # Store models with fold name  
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    
    
    