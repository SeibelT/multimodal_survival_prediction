import wandb
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from utils import survival_loss
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def simple_trainer(model,device,epochs,trainloader,testloader,bs,lr,alpha):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,)

    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-lr{lr}'

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
                
                loss = survival_loss(out,c,l,alpha)
                loss.backward() 
                optimizer.step()
                run.log({'loss': loss.item()},commit=True)
                
                
                
        
                        
            
            model.eval()
            with torch.no_grad():
                for  idx,(histo,gen,c,l) in enumerate(testloader):
                    histo.to(device)
                    gen.to(device)
                    c.to(device)
                    l.to(device)
                    out = model(histo,gen)
                    accuracy = ... # TODO 
                    loss = survival_loss(out,c,l,alpha)
                    run.log({'loss': loss.item()},commit=True)
                    run.log({'accuracy': accuracy, 'epoch': idx})    
        
