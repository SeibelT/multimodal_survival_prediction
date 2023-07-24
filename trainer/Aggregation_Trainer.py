import os
import torch
from utils.Aggregation_Utils import *


def store_checkpoint(epoch,model,optimizer,storepath,run_name):
    torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                os.path.join(storepath, f"{run_name}_{epoch}.pth"))


    
    
def Uni_Trainer_sweep(run,model,optimizer,criterion,trainloader,
                      testloader,bins,epochs,device,storepath,run_name,
                      l1_lambda,modality,batchsize
                      ):
    

    
    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        model.train()
        
        #init counter
        out_all =torch.empty(size=(0,bins),device='cpu')      
        c_all = torch.empty(size=(0,),device='cpu').to(torch.int16)
        l_all = torch.empty(size=(0,),device='cpu').to(torch.int16)
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
            elif modality=="hist_attention":
                weights = torch.cat([x.flatten() for x in model.Encoder.parameters()]) 
            
            loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1)
            
            loss.backward() 
            optimizer.step()
            
            runningloss += loss.item()
            
            #add to counters
            out_all= torch.cat((out_all,out),dim=0)
            l_all = torch.cat((l_all,l),dim=0)
            c_all = torch.cat((c_all,c),dim=0)
            del out,l,c

        c_index_train = c_index(out_all,c_all,l_all)

        #init counter
        out_all_val =torch.empty(size=(0,bins),device='cpu')        
        l_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
        l_con_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
        c_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
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
                
                out_all_val= torch.cat((out_all_val,out),dim=0)
                l_all_val = torch.cat((l_all_val,l),dim=0)
                c_all_val = torch.cat((c_all_val,c),dim=0)
                l_con_all_val = torch.cat((l_con_all_val,l_con),dim=0)
                

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
                      l1_lambda,modality,batchsize
                      ):
    

    
    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        model.train()
        
        #init counter
        out_all =torch.empty(size=(0,bins),device='cpu')      
        c_all = torch.empty(size=(0,),device='cpu').to(torch.int16)
        l_all = torch.empty(size=(0,),device='cpu').to(torch.int16)
        runningloss = 0
        
        for idx,(x,y,c,l,_) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x,y)
            out = out.cpu()
            
            if modality =="Porpoise":
                weights1 = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
                weights2 = torch.cat([x.flatten() for x in model.Attn_Mil.parameters()]) 
            
                loss = criterion(out,c,l) + 0.5 * l1_lambda * (torch.norm(weights1,1)+torch.norm(weights2,1)).cpu()
            
                
            elif modality=="PrePorpoise":
                weights1 = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
                weights2 = torch.cat([x.flatten() for x in model.Encoder.parameters()]) 
            
                loss = criterion(out,c,l) + 0.5 * l1_lambda * (torch.norm(weights1,1)+torch.norm(weights2,1)).cpu()
                
            loss.backward() 
            optimizer.step()
            
            runningloss += loss.item()
            
            #add to counters
            out_all= torch.cat((out_all,out),dim=0)
            l_all = torch.cat((l_all,l),dim=0)
            c_all = torch.cat((c_all,c),dim=0)
            del out,l,c

        c_index_train = c_index(out_all,c_all,l_all)

        #init counter
        out_all_val =torch.empty(size=(0,bins),device='cpu')        
        l_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
        l_con_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
        c_all_val = torch.empty(size=(0,),device='cpu').to(torch.int16)
        val_rloss = 0

        model.eval()
        with torch.no_grad():
            for  idx,(x,y,c,l,l_con) in enumerate(testloader):
                x = x.to(device)
                y = y.to(device)
                out = model(x,y)
                out = out.cpu()
                #loss = criterion(out,l)  #CE loss
                loss = criterion(out,c,l)  # TODO add loss regularization 
                val_rloss += loss.item()
                
                out_all_val= torch.cat((out_all_val,out),dim=0)
                l_all_val = torch.cat((l_all_val,l),dim=0)
                c_all_val = torch.cat((c_all_val,c),dim=0)
                l_con_all_val = torch.cat((l_con_all_val,l_con),dim=0)
                

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

    
    # Store models with fold name if needed   
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
