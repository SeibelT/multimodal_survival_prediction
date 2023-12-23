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
                      valloader,bins,epochs,device,storepath,run_name,
                      l1_lambda,modality,testloader=None
                      ):

    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        #init counter
        out_all = [] 
        c_all= []    
        l_all = []   
        l_con_all=[]
        running_loss_train = 0
        
        model.train()
        for idx,(x,c,l,l_con) in enumerate(trainloader):
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = out.cpu()
            
            if modality =="gen":
                weights = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
                loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1)
            elif modality=="hist":
                weights = torch.cat([x.flatten() for x in model.AttMil.parameters()])
                loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1) 
            elif modality=="hist_attention":
                weights = torch.cat([x.flatten() for x in model.Encoder.parameters()]) 
                loss = criterion(out,c,l) + l1_lambda * torch.norm(weights.cpu(),1)
            else:
                loss = criterion(out,c,l)
                            
            loss.backward() 
            optimizer.step()
            
            running_loss_train += loss.item()
            
            #add to counters
            out_all.append(out.detach()) 
            l_all.append(l)     
            c_all.append(c)   
            l_con_all.append(l_con)  
            del out,l,c
            
        risk_all = risk_func(torch.cat(out_all,dim=0))
        l_all = torch.cat(l_all,dim=0).cpu().to(torch.int16)
        c_all = torch.cat(c_all,dim=0).cpu().to(torch.int16)
        l_con_all = torch.cat(l_con_all,dim=0)
        
        c_index_train = c_index(risk_all,c_all,l_con_all)

        if valloader is not None: 
            c_index_val,running_loss_val,km_values_val = eval_func(model,valloader,criterion,device,bins,"unimodal")
            c_index_val_all[epoch] = c_index_val
            
            wandbdict = {"epoch": epoch+1,
                        "train/runningloss": running_loss_train/len(trainloader),
                        "train/c_index":c_index_train,
                        'valid/runningloss': running_loss_val/len(valloader),
                        "valid/c_index":c_index_val,
                    }
        else:
            wandbdict = {"epoch": epoch+1,
                        "train/runningloss": running_loss_train/len(trainloader),
                        "train/c_index":c_index_train,
                        }

        if run is not None:
            run.log(wandbdict)
            
    model_weights = model.state_dict()
    
    if run is not None:
        
        if testloader is None: 
            run.log(dict(c_index_max_val=c_index_val_all.max(),c_index_last_val=c_index_val_all[-1],c_index_epoch_val=np.argmax(c_index_val_all)))
            KM_wandb(run,km_values_val[0],km_values_val[1],event_cont=km_values_val[2],risk_group=km_values_val[3] ,nbins = 30)
            return c_index_val_all,km_values_val[0],km_values_val[1],km_values_val[2],km_values_val[3]
        else:
            c_index_test,running_loss_test,km_values_test = eval_func(model,testloader,criterion,device,bins,"unimodal")
            KM_wandb(run,km_values_test[0],km_values_test[1],event_cont=km_values_test[2],risk_group=km_values_test[3],nbins = 30)
            wandbdict_test = {
                            "test/runningloss": running_loss_test/len(testloader),
                            "test/c_index":c_index_test,
                        }
            run.log(wandbdict_test)
            return None,km_values_test[0],km_values_test[1],km_values_test[2],km_values_test[3]
            
    else:
        if testloader is None:
            c_index_train,running_loss_train,km_values_train = eval_func(model,trainloader,criterion,device,bins,"unimodal")# in eval mode 
            return c_index_train,c_index_val,model_weights
        else:
            c_index_train,running_loss_train,km_values_train = eval_func(model,trainloader,criterion,device,bins,"unimodal")# in eval mode 
            c_index_test,running_loss_test,km_values_test = eval_func(model,testloader,criterion,device,bins,"unimodal")
            return c_index_train,c_index_val,c_index_test,model_weights

    """
    # Store models with fold name  
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    
    """

def MM_Trainer_sweep(run,model,optimizer,criterion,trainloader,
                      valloader,bins,epochs,device,storepath,run_name,
                      l1_lambda,modality,testloader=None
                      ):
    c_index_val_all = torch.zeros(size=(epochs,))
    for epoch in range(epochs):
        #init counter
        out_all = [] 
        c_all= []    
        l_all = []   
        l_con_all = []
        running_loss_train = 0
        model.train()
        for idx,(x,y,c,l,l_con) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x,y)
            out = out.cpu()
            
            if modality =="Porpoise":
                weights1 = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
                weights2 = torch.cat([x.flatten() for x in model.Attn_Mil.parameters()]) 
            
                loss = criterion(out,c,l) + 0.5 * l1_lambda * (torch.norm(weights1,1)+torch.norm(weights2,1)).cpu()
            
                
            elif modality in ["PrePorpoise","PrePorpoise_meanagg_attmil","PrePorpoise_meanagg"]:
                weights1 = torch.cat([x.flatten() for x in model.SNN.parameters()]) 
                weights2 = torch.cat([x.flatten() for x in model.Encoder.parameters()]) 
            
                loss = criterion(out,c,l) + 0.5 * l1_lambda * (torch.norm(weights1,1)+torch.norm(weights2,1)).cpu()
                
            loss.backward() 
            optimizer.step()
            
            running_loss_train += loss.item()
            
            #add to counters
            out_all.append(out.detach()) 
            l_all.append(l)     
            c_all.append(c) 
            l_con_all.append(l_con)    
            del out,l,c
            
        risk_all = risk_func(torch.cat(out_all,dim=0))
        l_all = torch.cat(l_all,dim=0).cpu().to(torch.int16)
        c_all = torch.cat(c_all,dim=0).cpu().to(torch.int16)
        l_con_all = torch.cat(l_con_all,dim=0)
        c_index_train = c_index(risk_all,c_all,l_con_all)

        if valloader is not None: 
            c_index_val,running_loss_val,km_values_val = eval_func(model,valloader,criterion,device,bins,"multimodal")
            c_index_val_all[epoch] = c_index_val
            
            wandbdict = {"epoch": epoch+1,
                        "train/runningloss": running_loss_train/len(trainloader),
                        "train/c_index":c_index_train,
                        'valid/runningloss': running_loss_val/len(valloader),
                        "valid/c_index":c_index_val,
                        }
        else:
            wandbdict = {"epoch": epoch+1,
                        "train/runningloss": running_loss_train/len(trainloader),
                        "train/c_index":c_index_train,
                        }
        if run is not None:
            run.log(wandbdict)

    

    model_weights = model.state_dict()
    
    if run is not None:
            
        if testloader is None: 
            run.log(dict(c_index_max_val=c_index_val_all.max(),c_index_last_val=c_index_val_all[-1],c_index_epoch_val=np.argmax(c_index_val_all)))
            KM_wandb(run,km_values_val[0],km_values_val[1],event_cont=km_values_val[2],risk_group=km_values_val[3],nbins = 30)
            return c_index_val_all,km_values_val[0],km_values_val[1],km_values_val[2],km_values_test[3]
            
        else:
            c_index_test,running_loss_test,km_values_test = eval_func(model,testloader,criterion,device,bins,"multimodal")
            KM_wandb(run,km_values_test[0],km_values_test[1],event_cont=km_values_test[2],risk_group=km_values_test[3],nbins = 30)
            wandbdict_test = {
                            "test/runningloss": running_loss_test/len(testloader),
                            "test/c_index":c_index_test,
                        }
            run.log(wandbdict_test)
            return None,km_values_test[0],km_values_test[1],km_values_test[2],km_values_test[3]
            
    else:
        if testloader is None:
            c_index_train,running_loss_train,km_values_train = eval_func(model,trainloader,criterion,device,bins,"multimodal")# in eval mode 
            return c_index_train,c_index_val,model_weights
        else:
            c_index_train,running_loss_train,km_values_train = eval_func(model,trainloader,criterion,device,bins,"multimodal")# in eval mode 
            c_index_test,running_loss_test,km_values_test = eval_func(model,testloader,criterion,device,bins,"multimodal")
            return c_index_train,c_index_val,c_index_test,model_weights
        
    """
    # Store models with fold name if needed   
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    
    torch.save({
                'model': model.state_dict(),
                },
                os.path.join(storepath, f"{run_name}.pth"))
    """
    
def eval_func(model,loader,criterion,device,bins,modality_input):
    #uni
    #init counter
    out_all = []   
    l_all = []
    l_con_all = []
    c_all = []
    running_loss = 0

    model.eval()
    if modality_input == "unimodal":
        with torch.no_grad():
            for  idx,(x,c,l,l_con) in enumerate(loader):
                x = x.to(device)
                out = model(x)
                out = out.cpu()
                
                if criterion is not None:
                    loss = criterion(out,c,l) 
                    running_loss += loss.item()                
                out_all.append(out) 
                l_all.append(l) 
                c_all.append(c)
                l_con_all.append(l_con) 
                
    elif modality_input == "multimodal":
        with torch.no_grad():
            for  idx,(x,y,c,l,l_con) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                out = model(x,y)
                out = out.cpu()
                
                if criterion is not None:
                    loss = criterion(out,c,l)  
                    running_loss += loss.item()
                    
                out_all.append(out) 
                l_all.append(l)
                c_all.append(c)
                l_con_all.append(l_con)

    risk_all = risk_func(torch.cat(out_all,dim=0))
    l_all = torch.cat(l_all,dim=0)
    c_all = torch.cat(c_all,dim=0)
    l_con_all = torch.cat(l_con_all,dim=0)
    
    c_index_out = c_index(risk_all,c_all,l_con_all)    
    risk_group = risk_all>risk_all.median()
    return c_index_out,running_loss,[risk_all,c_all,l_con_all,risk_group]