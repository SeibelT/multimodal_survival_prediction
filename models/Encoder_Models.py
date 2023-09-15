import torch
from torch import nn
from torchvision.models import resnet18

import pytorch_lightning as pl
from utils.Encoder_Utils import c_index,Survival_Loss
from torchmetrics import Accuracy
from models.mae_models.models_mae_modified import mae_vit_tiny_patch16
import torch
import h5py


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
            
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
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
    def __init__(self,lr,nbins,alpha,ckpt_path=None,ffcv=False,encode_gen=True):
        super().__init__()
        self.lr = lr
        self.nbins = nbins
        self.mask_ratio = 0.75
        self.ids_shuffle = None
        self.save_hyperparameters()
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #load weights
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        
        
        self.classification_head =  Classifier_Head(2*192,d_hidden=256,t_bins=nbins)
        
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        
        self.y_encoder = SNN(d=20971,d_out = 192,activation="SELU")
        #prediction
        self.encode_gen = encode_gen
        
    def forward(self, x,y):
        y = self.y_encoder(y)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y.unsqueeze(1), self.ids_shuffle)
        latent_x,latent_y = torch.split(latent,split_size_or_sections=[latent.size(1)-1,1],dim=1)
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        conc_latent = torch.cat((torch.mean(latent_x,dim=1),latent_y.squeeze(1)),dim=1)
        surv_logits = self.classification_head(conc_latent)
        return pred,mask,surv_logits

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_surv = self.criterion(surv_logits,censorship,label)
        
        self.log("train_MAEloss", loss_MAE)
        self.log("train_Survloss",loss_surv)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE+loss_surv
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
            self.log(f"{stage}surv_loss", loss_surv,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    
    def oldpredstp(self, batch, batch_idx,ids_shuffle,mask_ratio=0.8):
        with torch.no_grad():
            if self.encode_gen:
                x,y, coords = batch
                y = self.y_encoder(y)
                latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y.unsqueeze(1), ids_shuffle)
                latent_x,latent_y = torch.split(latent,split_size_or_sections=[latent.size(1)-1,1],dim=1)
                conc_latent = torch.cat((torch.mean(latent_x,dim=1),latent_y.squeeze(1)),dim=1)
                return latent_x,conc_latent,coords,ids_restore
            else:
                x,y, coords = batch
                latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, ids_shuffle)
                #latent = torch.mean(latent,dim=1)
                return latent,coords,ids_restore
        
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        
        if self.encode_gen:
            y = self.y_encoder(y)
            latent_xy, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y.unsqueeze(1), self.ids_shuffle)
            latent_x,latent_y = torch.split(latent_xy,split_size_or_sections=[latent_xy.size(1)-1,1],dim=1)
            conc_latent = torch.cat((torch.mean(latent_x,dim=1),latent_y.squeeze(1)),dim=1)
            latent = conc_latent
            
        else:
            
            latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, self.ids_shuffle)
            latent = torch.mean(latent,dim=1)
        
        return (latent)
            
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            )
        
        return {"optimizer": optimizer}


class VitTiny(pl.LightningModule):
    def __init__(self,lr,ckpt_path=None,ffcv=False,**kwargs):
        super().__init__()
        self.lr = lr
        self.mask_ratio = 0.75
        self.ids_shuffle = None
        self.save_hyperparameters()
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #load weights
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        self.y_empty  = torch.rand(size=(1,0,192))
        
    def forward(self, x):
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        pred = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred,mask

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask = self(hist_tile)
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        self.log("train_MAEloss", loss_MAE)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask = self(hist_tile)
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
        
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, self.ids_shuffle)
        latent = torch.mean(latent,dim=1)
        return (latent)
            
    def encodedecode(self,batch,ids_shuffle,mask_ratio):
        x = batch
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, ids_shuffle)
        pred = self.model.forward_decoder(latent, ids_restore)
        loss_MAE = self.model.forward_loss(x, pred, mask)
        pred = self.model.unpatchify(pred)
        return latent,pred,loss_MAE
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            )
        
        return {"optimizer": optimizer}



class VitTiny_freeze(pl.LightningModule):
    def __init__(self,lr,ckpt_path=None,ffcv=False):
        super().__init__()
        self.lr = lr
        self.mask_ratio = 0.75
        self.ids_shuffle = None
        self.save_hyperparameters()
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #load weights
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
            for idx,block in enumerate(self.model.blocks):
                if idx<=(len(self.model.blocks)-2):
                    block.requires_grad=False
            
        self.model.cls_token.requires_grad=False
        self.model.patch_embed.proj.weight.requires_grad=False
        self.model.patch_embed.proj.bias.requires_grad=False
        for idx,block in enumerate(self.model.blocks):
            if idx<=(len(self.model.blocks)-2): #all, except last block
                for names,parms in block.named_parameters():
                    parms.requires_grad = False
        

        self.y_empty  = torch.rand(size=(1,0,192))
        
    def forward(self, x):
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        pred = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred,mask

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask = self(hist_tile)
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        self.log("train_MAEloss", loss_MAE)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask = self(hist_tile)
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
        
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, self.ids_shuffle)
        latent = torch.mean(latent,dim=1)
        return (latent)
            
    def encodedecode(self,batch,ids_shuffle,mask_ratio):
        x = batch
        y = self.y_empty.to(x.device).repeat(x.size(0),1,1)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y, ids_shuffle)
        pred = self.model.forward_decoder(latent, ids_restore)
        loss_MAE = self.model.forward_loss(x, pred, mask)
        pred = self.model.unpatchify(pred)
        return latent,pred,loss_MAE
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            )
        
        return {"optimizer": optimizer}



class SNN(nn.Module):
    """ Implementation of SNN as described in 'Pan-cancer integrative histology-genomic analysis via multimodal deep learning' by R.Chen et al 
    https://pubmed.ncbi.nlm.nih.gov/35944502/ 
    
    Variables:
    d : dimension of molecular vector
    d_out : dimnesion of embedded output vector 
    """
    def __init__(self,d : int,d_out : int = 32,activation="SELU"):
        super(SNN,self).__init__()
        
        
        self.lin1 = nn.Linear(d,256)
        

        self.lin2 = nn.Linear(256,256)
        self.alphadropout1 = nn.AlphaDropout(p=0.5)
        self.alphadropout2 = nn.AlphaDropout(p=0.5)
        self.fc = nn.Linear(256,d_out)
        if activation=="SELU":
            torch.nn.init.normal_(self.lin1.weight, mean=0, std=1/d**0.5)
            torch.nn.init.normal_(self.lin2.weight, mean=0, std=1/256**0.5)
            torch.nn.init.normal_(self.fc.weight, mean=0, std=1/256**0.5)
            self.selu1 = nn.SELU()
            self.selu2 = nn.SELU()
            self.selu3 = nn.SELU()
        elif activation=="RELU":
            torch.nn.init.kaiming_normal_(self.lin1.weight)
            torch.nn.init.kaiming_normal_(self.lin2.weight)
            torch.nn.init.kaiming_normal_(self.fc.weight)
            self.selu1 = nn.ReLU()
            self.selu2 = nn.ReLU()
            self.selu3 = nn.ReLU()
        
        elif activation=="GELU":
            torch.nn.init.kaiming_normal_(self.lin1.weight)
            torch.nn.init.kaiming_normal_(self.lin2.weight)
            torch.nn.init.kaiming_normal_(self.fc.weight)
            self.selu1 = nn.GELU()
            self.selu2 = nn.GELU()
            self.selu3 = nn.GELU()

    def forward(self,x):

        x = self.alphadropout1(self.selu1(self.lin1(x)))
        x = self.alphadropout2(self.selu2(self.lin2(x)))
        x = self.fc(x)
        return self.selu3(x)