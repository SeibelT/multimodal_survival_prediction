import torch
from torch import nn
from torchvision.models import resnet18
from utils.Aggregation_Utils import Survival_Loss
import pytorch_lightning as pl

from torchmetrics import Accuracy
from sksurv.metrics import concordance_index_censored
import torch
# Your custom model


class Resnet18Surv(pl.LightningModule):
    def __init__(self,lr_init,lr_unfrozen,nbins,alpha,tsteps,unfreeze_backbone_epoch):
        super().__init__()
        self.lr_init = lr_init
        self.lr_unfrozen = lr_unfrozen
        self.lr = lr_init
        self.tsteps = tsteps
        self.nbins = nbins
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self.frozen_flag = True
        self.save_hyperparameters()
        # Model
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()
        self.classification_head =  Classifier_Head(512,d_hidden=256,t_bins=nbins)
        # Loss
        self.criterion = Survival_Loss(alpha)
        
        #C-index
        self.logits_all =[]
        self.l_all = []
        self.l_con_all =[]
        self.c_all = []
        
        
    def forward(self, x):
        out = self.resnet18(x)
        out = self.classification_head(out)
        return out

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        logits = self(hist_tile)
        loss = self.criterion(logits,censorship,label)
        self.log("train_loss", loss)
        self.log("learning_rate",self.hparams.lr)
        
        
        return loss
    def on_epoch_end(self):
        self.log("Backbone_frozen", self.frozen_flag)
        if self.trainer.current_epoch == self.unfreeze_backbone_epoch:
            self.frozen_flag = True
            self.lr = self.lr_unfrozen
            ?# TODO reconfig optimiizer with all weights and new learning rate 
            # check if returns issues with StochasticWeightAveraging
            
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        logits = self(hist_tile)
        loss = self.criterion(logits,censorship,label)

        self.logits_all.append(logits)
        self.l_all.append(label)
        self.c_all.append(censorship)
        self.l_con_all.append(label_cont)
        
        
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    def on_validation_epoch_end(self):
        h = nn.Sigmoid()(torch.cat(self.logits_all,dim=0))
        S = torch.cumprod(1-h,dim = -1)
        risk = -S.sum(dim=1) 
        notc = (1-torch.cat(self.c_all,dim=0)).cpu().numpy().astype(bool)
        try:
            c_index = concordance_index_censored(notc, torch.cat(self.l_all,dim=0).cpu(),risk.cpu())
            #log
            self.log(f"c_index",c_index, prog_bar=True)
        except:
            print("C index problems,probably all samples are censored")
            c_index = None
        #log
        self.log(f"c_index",c_index, prog_bar=True)
        
        
        
        #free memory
        self.logits_all.clear()
        self.l_all.clear()
        self.l_con_all.clear()
        self.c_all.clear()
        del h,S,risk,notc
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            )
        
        return {"optimizer": optimizer}



class Classifier_Head(nn.Module):
    def __init__(self,outsize,d_hidden=256,t_bins=4):
        super(Classifier_Head,self).__init__()

        self.linear1 = nn.Linear(outsize,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.activ1 = nn.ReLU()
        self.linear2  = nn.Linear(d_hidden,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.activ2 = nn.ReLU()
        self.fc = nn.Linear(d_hidden,t_bins) 
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.activ1(self.linear1(x))
        x = self.activ2(self.linear2(x))
        return self.fc(x)
    
    

def c_index(out_all,c_all,l_all):
    """
    Variables
    out_all : FloatTensor must be of shape = (N,4)  predicted logits of model 
    c_all : IntTensor must be of shape = (N,) 
    l_all IntTensor must be of shape = (N,)

    Outputs the c-index score 
    """
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