import torch
from torch import nn
from torchvision.models import resnet18
from utils.Aggregation_Utils import Survival_Loss
import pytorch_lightning as pl
from utils.Encoder_Utils import c_index
from torchmetrics import Accuracy

import torch
# Your custom model


class Resnet18Surv(pl.LightningModule):
    def __init__(self,lr,nbins,alpha):
        super().__init__()
        self.lr = lr
        self.nbins = nbins
        self.frozen_flag = True
        
        # Model
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()
        self.classification_head =  Classifier_Head(512,d_hidden=256,t_bins=nbins)
        # Loss
        self.criterion = Survival_Loss(alpha)
        
        self.save_hyperparameters()
        
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
        c_ind = c_index(self.logits_all,self.c_all,self.l_all)
        #log
        self.log(f"c_index",c_ind, prog_bar=True)
        
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
    
    



class SupViTSurv(pl.LightningModule):
    def __init__(self,lr,nbins,alpha,mae_training,supervised_surv,multimodal):
        super().__init__()
        self.lr = lr
        self.nbins = nbins
        self.frozen_flag = True
        
        # Model
        if mae_training:
            self.encoder = ...
            self.decoder = ...
            
            
        if supervised_surv:
            self.classification_head =  Classifier_Head(512,d_hidden=256,t_bins=nbins)
            
            
        if multimodal:
            self.aggregation = ...    
            
            
        # Loss
        self.criterion_MAE = nn.MSELoss()
        self.criterionSurv = Survival_Loss(alpha)
        
        self.save_hyperparameters()
        
        #C-index
        self.logits_all =[]
        self.l_all = []
        self.l_con_all =[]
        self.c_all = []
        
        
    def forward(self, x,y):
        x = self.img2seq(x)
        if self.hparams.mae_training:
            unmasked,masked,masktokens,unmasked_idx,masked_idx = self.masking(x)
            x_enc, x_idx = unmasked, unmasked_idx
        else:
            x_enc, x_idx = x, self.regular_idx
        
        if self.hparams.multimodal: 
            x_enc,y_enc = self.split(self.Encoder(torch.cat([self.gen_embedder(y),self.Encoder_Embedding(x_enc)+self.Enc_Pos_Embedding],dim=1))) #check if correct dim
        else:
            x_enc = self.Encoder(self.Encoder_Embedding(x_enc)+self.Enc_Pos_Embedding)
            
        if self.hparams.mae_training:
            decoded = self.Decoder(torch.stack([self.Decoder_Embedding(x_enc) +self.Dec_Pos_Embedding(x_idx),masktokens+self.Dec_Pos_Embedding(masked_idx)],dim=1))#check right dim
            unmasked_out,masked_out = torch.split(decoded)# split via idx 
            
        
        if self.hparams.supervised_surv:
            if self.hparams.multimodal: 
                logits = self.classification_head(torch.stack([torch.mean(x_enc,y_enc)],dim=1))
                return 
            else:
                
            
            
        
        
            
        
        #Encoding    
        
        
        if 
        
        
        
         
        out = self.decoder(x)
        return x,out

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        if self.hparams.mae_training:
            
        if self.hparams.multimodal:
            
        
        
        logits = self(hist_tile)
        loss = self.criterion(logits,censorship,label)
        self.log("train_loss", loss)
        self.log("learning_rate",self.hparams.lr)
        return loss
    
            
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
        c_ind = c_index(self.logits_all,self.c_all,self.l_all)
        #log
        self.log(f"c_index",c_ind, prog_bar=True)
        
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




