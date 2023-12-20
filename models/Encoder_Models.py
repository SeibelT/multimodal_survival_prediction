import torch
from torch import nn
from torchvision.models import resnet18,resnet50,ResNet50_Weights
import pytorch_lightning as pl
from utils.Encoder_Utils import c_index,Survival_Loss
from torchmetrics import Accuracy
from models.mae_models.models_mae_modified import mae_vit_tiny_patch16
import torch
import h5py
import copy
import os
import psutil
import gc
class Classifier_Head(nn.Module):
    """Survival Head"""
    def __init__(self,outsize,p_dropout_head,d_hidden=256,t_bins=4,):
        super(Classifier_Head,self).__init__()
        
        self.dropout1 = nn.Dropout(p=p_dropout_head)
        self.dropout2 = nn.Dropout(p=p_dropout_head)
        
        self.linear1 = nn.Linear(outsize,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.activ1 = nn.ReLU()
        self.linear2  = nn.Linear(d_hidden,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.activ2 = nn.ReLU()
        self.fc = nn.Linear(d_hidden,t_bins) 
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.dropout1(self.activ1(self.linear1(x)))
        x = self.dropout2(self.activ2(self.linear2(x)))
        return self.fc(x)
    
    

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
    
    
def Mean_Aggregation(latent_x,latent_y):
    return torch.mean(torch.cat((latent_x,latent_y),dim=1),dim=1)
def Concat_Mean_Aggregation(latent_x,latent_y):
    return  torch.cat((torch.mean(latent_x,dim=1),latent_y.squeeze(1)),dim=1)
        
    
    

class SupViTSurv(pl.LightningModule):
    def __init__(self,lr,nbins,alpha,aggregation_func,d_gen=20971 ,ckpt_path=None,ffcv=False,encode_gen=True,p_dropout_head=0):
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
        
        
        f_class = 1 if aggregation_func=="Mean_Aggregation" else 2 if aggregation_func=="Concat_Mean_Aggregation" else None
        self.classification_head =  Classifier_Head(f_class*192,d_hidden=128*f_class,t_bins=nbins,p_dropout_head=p_dropout_head)
        
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        self.aggregation = Mean_Aggregation if aggregation_func=="Mean_Aggregation" else Concat_Mean_Aggregation if aggregation_func=="Concat_Mean_Aggregation" else None
        self.y_encoder = SNN(d=d_gen,d_out = 192,activation="SELU")
        #prediction
        self.encode_gen = encode_gen
        
    def forward(self, x,y):
        y = self.y_encoder(y)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y.unsqueeze(1), self.ids_shuffle)
        latent_x,latent_y = torch.split(latent,split_size_or_sections=[latent.size(1)-1,1],dim=1)
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        conc_latent = self.aggregation(latent_x,latent_y)
        surv_logits = self.classification_head(conc_latent)
        return pred,mask,surv_logits

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_Surv = self.criterion(surv_logits,censorship,label)
        
        self.log("train_MAEloss", loss_MAE)
        self.log("train_Survloss",loss_Surv)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE+loss_Surv
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
            self.log(f"{stage}surv_loss", loss_surv,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        
        if self.encode_gen:
            y = self.y_encoder(y)
            latent_xy, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y.unsqueeze(1), self.ids_shuffle)
            latent_x,latent_y = torch.split(latent_xy,split_size_or_sections=[latent_xy.size(1)-1,1],dim=1)
            conc_latent = self.aggregation(latent_x,latent_y)
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





class SupViTSurvNoMAE(pl.LightningModule):
    def __init__(self,lr,aggregation_func,genomics,nbins,alpha,mask_ratio,encode_gen,d_gen=20971 ,ckpt_path=None,ffcv=False,p_dropout_head=0,**kwargs):
        super().__init__()
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.ids_shuffle = None
        self.save_hyperparameters()
        #Ensure
        assert aggregation_func in ["Mean_Aggregation","Concat_Mean_Aggregation"]
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #Survival
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        #Load weights
        if ckpt_path is not None:
            """Load Imagenet1K pretraining"""
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        #Genmomics Modality
        if genomics:
            g_enc_dim = 192
            self.y_encoder = self.encoded_y
            self.snn = SNN(d=d_gen,d_out = 192,activation="SELU")
            self.split_func = self.split_mm
        else:
            self.y_encoder = self.empty_y
            self.split_func = self.split_um
            g_enc_dim=0
        #Aggregationtype for Survival-Head and Encoding
        if aggregation_func=="Mean_Aggregation":
            self.classification_head =  Classifier_Head(192,d_hidden=128,t_bins=nbins,p_dropout_head=p_dropout_head)
            self.aggregation = Mean_Aggregation
        elif aggregation_func=="Concat_Mean_Aggregation":
            self.classification_head =  Classifier_Head(192+g_enc_dim,d_hidden=128,t_bins=nbins,p_dropout_head=p_dropout_head)
            self.aggregation = Concat_Mean_Aggregation
        #Prediction
        self.encode_gen = encode_gen
        
    def empty_y(self,y):
        """helper function, returns empty vector"""
        B,d = y.size()
        return torch.rand(size=(B,0,192))
    def encoded_y(self,y):
        y = self.snn(y)
        return y.unsqueeze(1)
    def split_mm(self,x):
        """helper function, split multimodal encoding"""
        latent_x,latent_y = torch.split(x,split_size_or_sections=[x.size(1)-1,1],dim=1)
        return latent_x,latent_y
    def split_um(self,x):
        """helper function, split unimodal encoding"""
        B,_,d = x.size()
        u = torch.rand(size=(B,0,192))
        return x,u
    
    def forward(self,x,y):
        y = self.y_encoder(y).to(x.device)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        latent_x,latent_y = self.split_func(latent)
        conc_latent = self.aggregation(latent_x,latent_y.to(latent_x.device))
        return conc_latent 

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        conc_latent = self(hist_tile,gen)
        surv_logits = self.classification_head(conc_latent)
        loss_Surv = self.criterion(surv_logits,censorship,label)
        
        self.log("train_Survloss",loss_Surv)
        self.log("learning_rate",self.hparams.lr)
        return loss_Surv
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        conc_latent = self(hist_tile,gen)
        surv_logits = self.classification_head(conc_latent)
        loss_surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}surv_loss", loss_surv,sync_dist=True)
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        if self.encode_gen:
            latent = self(x,y)
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
    
    
class Resnet18Surv(pl.LightningModule):
    def __init__(self,lr,nbins,alpha,p_dropout_head=0):
        super().__init__()
        self.lr = lr
        self.nbins = nbins
        self.frozen_flag = True
        
        # Model
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()
        self.classification_head =  Classifier_Head(512,d_hidden=256,t_bins=nbins,p_dropout_head=p_dropout_head)
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




class MultiSupViTSurv(pl.LightningModule):
    def __init__(self,lr,nbins,n_neighbours,alpha,aggregation_func,d_gen=20971 ,ckpt_path=None,ffcv=False,encode_gen=True,p_dropout_head=0):
        super().__init__()
        #self.automatic_optimization = False
        self.lr = lr
        self.nbins = nbins
        self.mask_ratio = 0.75
        self.ids_shuffle = None
        self.n_tiles = n_neighbours+1
        self.save_hyperparameters()
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #load weights
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        
        
        f_class = 1 if aggregation_func=="Mean_Aggregation" else 2 if aggregation_func=="Concat_Mean_Aggregation" else None
        self.classification_head =  Classifier_Head(f_class*192,d_hidden=128*f_class,t_bins=nbins,p_dropout_head=p_dropout_head)
        
        
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        self.aggregation = Mean_Aggregation if aggregation_func=="Mean_Aggregation" else Concat_Mean_Aggregation if aggregation_func=="Concat_Mean_Aggregation" else None
        self.y_encoder = SNN(d=d_gen,d_out = 192,activation="SELU")
        #prediction
        self.encode_gen = encode_gen
        self.lin_encs =  nn.ModuleList([copy.deepcopy(self.model.patch_embed) for i in range(n_neighbours)]+[self.model.patch_embed])
        assert n_neighbours==3
        
    def forward(self, x,y):
        y = self.y_encoder(y).unsqueeze(1)
        x = self.lin_encs[0](x) + self.model.pos_embed[:, 1:, :] 
        
        x_enc, mask, ids_restore,_ = self.model.random_masking(x, self.mask_ratio, self.ids_shuffle)
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]

        
        cls_tokens = cls_token.expand(y.size(0), -1, -1)
        x_enc = torch.cat((cls_tokens, x_enc,y), dim=1)

        #encode sequence 
        for blk in self.model.blocks:
            x_enc = blk(x_enc)
        x_enc = self.model.norm(x_enc)
        
        #separate y_encoding_images
        cls_token_enc,latent_x,latent_y = torch.split(x_enc,split_size_or_sections=[1,x_enc.size(1)-2,1],dim=1)
        #surv loss 
        conc_latent = torch.mean(torch.cat((latent_x,latent_y),dim=1),dim=1)
        surv_logits = self.classification_head(conc_latent)
        #MAE decoder
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        conc_latent = self.aggregation(latent_x,latent_y)
        surv_logits = self.classification_head(conc_latent)
        return pred,mask,surv_logits

    
    
    def training_step(self, batch, batch_idx):
        nn_tiles,gen, censorship, label = batch
        
        #linear embedding, positional encoding, masking
        x_lin = [lin_encoding(x.squeeze(1))+ self.model.pos_embed[:, 1:, :]  for lin_encoding,x in zip(self.lin_encs,torch.split(nn_tiles,split_size_or_sections = 1,dim=1))]
        image_function_outputs = [self.model.random_masking(x, self.mask_ratio, self.ids_shuffle) for x in x_lin]
        x_enc, masks, ids_restore_list,_ = zip(*image_function_outputs)
        del _
        
        
        # concat to single sequence with cls token and gen encoding 
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        y = self.y_encoder(gen).unsqueeze(1)
        
        cls_tokens = cls_token.expand(gen.size(0), -1, -1)
        x_enc = torch.cat(x_enc,dim=1)
        x_enc = torch.cat((cls_tokens, x_enc,y), dim=1)

        #encode sequence 
        for blk in self.model.blocks:
            x_enc = blk(x_enc)
        x_enc = self.model.norm(x_enc)

        
        
        #separate y_encoding_images
        cls_token_enc,latent_x,latent_y = torch.split(x_enc,split_size_or_sections=[1,x_enc.size(1)-2,1],dim=1)
        #surv loss 
        conc_latent = torch.mean(torch.cat((latent_x,latent_y),dim=1),dim=1)
        surv_logits = self.classification_head(conc_latent)
        loss_Surv = self.criterion(surv_logits,censorship,label)
        self.log("train_Survloss",loss_Surv.detach().item(),on_epoch=False)
        
        #decode
        latent_x = self.model.decoder_embed(latent_x)#ok
        cls_token_dec = self.model.decoder_embed(cls_token_enc).repeat(1,self.n_tiles,1).reshape(cls_token_enc.size(0)*self.n_tiles,1,latent_x.size(2))# B,1,d -> B,4,D -> B*4,1,d
        #stack all images into batch 
        latent_x=latent_x.reshape(latent_x.size(0)*self.n_tiles,int(latent_x.size(1)//self.n_tiles),latent_x.size(2))#B,4*l_seq,d -> B*4,l_seq,d
        masks = torch.stack(masks,dim=1).reshape(latent_x.size(0),196)# [img_mask1,img_mask2,..,imgmask4]->B,4,196 -> B*4,196
        ids_restore_list = torch.stack(ids_restore_list,dim=1).reshape(latent_x.size(0),196) # same as mask 
        #decode_steps
        mask_tokens = self.model.mask_token.repeat(latent_x.shape[0], ids_restore_list.shape[1] - latent_x.shape[1], 1) # masked w masktoken 
        latent_x = torch.cat([latent_x, mask_tokens], dim=1) # add to sequence 
        latent_x = torch.gather(latent_x, dim=1, index=ids_restore_list.unsqueeze(-1).repeat(1, 1, latent_x.shape[2]))  # unshuffle
        latent_x = torch.cat([cls_token_dec, latent_x], dim=1)
        
        # add pos embed
        latent_x = latent_x + self.model.decoder_pos_embed
        # apply Transformer blocks for decoding
        for blk in self.model.decoder_blocks:
            latent_x = blk(latent_x)
        latent_x = self.model.decoder_norm(latent_x)
        # predictor projection
        latent_x = self.model.decoder_pred(latent_x)
        latent_x = latent_x[:, 1:, :]# remove cls token
        
        loss_MAE = self.model.forward_loss(nn_tiles.reshape(latent_x.size(0),3,224,224), latent_x, masks)
        self.log(f"train_MAEloss", loss_MAE.detach().item(),on_epoch=False)
        print(f"RAM",psutil.virtual_memory().percent)
        #####debug
        
        
        return loss_Surv + loss_MAE
        ############
        
        opt = self.optimizers()
        N = 1
        # scale losses by 1/N (for N batches of gradient accumulation)
        loss = (loss_Surv+4*loss_MAE) / N
        self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            del batch,nn_tiles,x_lin,image_function_outputs,x_enc, masks, ids_restore_list,latent_x,loss_MAE,loss_Surv
            del blk,cls_token,cls_token_dec,cls_token_enc,cls_tokens,conc_latent,gen,latent_y,loss,mask_tokens,y,surv_logits,label,censorship
            gc.collect()
            #############
            #############
            #############
        
        
        
    
        
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
            self.log(f"{stage}surv_loss", loss_surv,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        
        if self.encode_gen:
            y = self.y_encoder(y).unsqueeze(1)
            x = self.lin_encs[0](x) + self.model.pos_embed[:, 1:, :] 
            
            x_enc, mask, ids_restore,_ = self.model.random_masking(x, mask_ratio, self.ids_shuffle)
            cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]

            cls_tokens = cls_token.expand(y.size(0), -1, -1)
            x_enc = torch.cat((cls_tokens, x_enc,y), dim=1)
            #encode sequence 
            for blk in self.model.blocks:
                x_enc = blk(x_enc)
            x_enc = self.model.norm(x_enc)
            #separate y_encoding_images
            cls_token_enc,latent_x,latent_y = torch.split(x_enc,split_size_or_sections=[1,x_enc.size(1)-2,1],dim=1)
            latent = torch.mean(torch.cat((latent_x,latent_y),dim=1),dim=1)
            
            
        else:
        
            x = self.lin_encs[0](x) + self.model.pos_embed[:, 1:, :] 
            x_enc, mask, ids_restore,_ = self.model.random_masking(x, mask_ratio, self.ids_shuffle)
            cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]

            cls_tokens = cls_token.expand(x.size(0), -1, -1)
            x_enc = torch.cat((cls_tokens, x_enc), dim=1)
            #encode sequence 
            for blk in self.model.blocks:
                x_enc = blk(x_enc)
            x_enc = self.model.norm(x_enc)
            #separate y_encoding_images
            cls_token_enc,latent_x = torch.split(x_enc,split_size_or_sections=[1,x_enc.size(1)-1],dim=1)
            latent = torch.mean(latent_x,dim=1)
            
        
        return (latent)
            
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            )
        
        return {"optimizer": self.optimizer}








class SupViTSurv2(pl.LightningModule):
    """experimental:loss multiplied"""
    def __init__(self,lr,nbins,alpha,aggregation_func,d_gen=20971 ,ckpt_path=None,ffcv=False,encode_gen=True,p_dropout_head=0):
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
        
        
        f_class = 1 if aggregation_func=="Mean_Aggregation" else 2 if aggregation_func=="Concat_Mean_Aggregation" else None
        self.classification_head =  Classifier_Head(f_class*192,d_hidden=128*f_class,t_bins=nbins,p_dropout_head=p_dropout_head)
        
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        self.aggregation = Mean_Aggregation if aggregation_func=="Mean_Aggregation" else Concat_Mean_Aggregation if aggregation_func=="Concat_Mean_Aggregation" else None
        self.y_encoder = SNN(d=d_gen,d_out = 192,activation="SELU")
        #prediction
        self.encode_gen = encode_gen
        
    def forward(self, x,y):
        y = self.y_encoder(y)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y.unsqueeze(1), self.ids_shuffle)
        latent_x,latent_y = torch.split(latent,split_size_or_sections=[latent.size(1)-1,1],dim=1)
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        conc_latent = self.aggregation(latent_x,latent_y)
        surv_logits = self.classification_head(conc_latent)
        return pred,mask,surv_logits

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_Surv = self.criterion(surv_logits,censorship,label)
        
        self.log("train_MAEloss", loss_MAE)
        self.log("train_Survloss",loss_Surv)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE*loss_Surv
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        pred,mask,surv_logits = self(hist_tile,gen)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
            self.log(f"{stage}surv_loss", loss_surv,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        
        if self.encode_gen:
            y = self.y_encoder(y)
            latent_xy, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, mask_ratio,y.unsqueeze(1), self.ids_shuffle)
            latent_x,latent_y = torch.split(latent_xy,split_size_or_sections=[latent_xy.size(1)-1,1],dim=1)
            conc_latent = self.aggregation(latent_x,latent_y)
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





class ViTMAEtiny_gen(pl.LightningModule):
    def __init__(self,lr,aggregation_func,genomics,nbins,alpha,mask_ratio,encode_gen,d_gen=20971 ,ckpt_path=None,ffcv=False,p_dropout_head=0,**kwargs):
        super().__init__()
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.ids_shuffle = None
        self.save_hyperparameters()
        #Ensure
        assert aggregation_func in ["Mean_Aggregation","Concat_Mean_Aggregation"]
        #ViT 
        self.model = mae_vit_tiny_patch16()
        
        #Load weights
        if ckpt_path is not None:
            """Load Imagenet1K pretraining"""
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        #Genmomics Modality
        if genomics:
            g_enc_dim = 192
            self.y_encoder = self.encoded_y
            self.snn = SNN(d=d_gen,d_out = 192,activation="SELU")
            self.split_func = self.split_mm
        else:
            self.y_encoder = self.empty_y
            self.split_func = self.split_um
            
        #Aggregationtype for Survival-Head and Encoding
        if aggregation_func=="Mean_Aggregation":
            self.aggregation = Mean_Aggregation
        elif aggregation_func=="Concat_Mean_Aggregation":
            self.aggregation = Concat_Mean_Aggregation
        #Prediction
        self.encode_gen = encode_gen
        
    def empty_y(self,y):
        """helper function, returns empty vector"""
        B = y.size(0)
        return torch.rand(size=(B,0,192))
    def encoded_y(self,y):
        y = self.snn(y)
        return y.unsqueeze(1)
    def split_mm(self,x):
        """helper function, split multimodal encoding"""
        latent_x,latent_y = torch.split(x,split_size_or_sections=[x.size(1)-1,1],dim=1)
        return latent_x,latent_y
    def split_um(self,x):
        """helper function, split unimodal encoding"""
        B,_,d = x.size()
        u = torch.rand(size=(B,0,192))
        return x,u
    
    def forward(self,x,y):
        y = self.y_encoder(y).to(x.device)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        latent_x,latent_y = self.split_func(latent)
        return latent_x,latent_y,mask,ids_restore

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        latent_x,latent_y,mask,ids_restore = self(hist_tile,gen)
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        
        self.log("train_MAEloss", loss_MAE)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        latent_x,latent_y,mask,ids_restore = self(hist_tile,gen)
        pred = self.model.forward_decoder(latent_x, ids_restore)  # [N, L, p*p*3]
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
            
        if stage:
            self.log(f"{stage}_MAEloss", loss_MAE,sync_dist=True)
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        if self.encode_gen:
            latent_x,latent_y,mask,ids_restore = self(x,y)
            latent = self.aggregation(latent_x,latent_y.to(latent_x.device))
        else:
            y = self.empty_y(x).to(x.device)
            latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
            latent_x,latent_y = self.split_func(latent)
            latent = self.aggregation(latent_x,latent_y.to(latent_x.device))
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
    




class SupViTSurv_nogen(pl.LightningModule):
    def __init__(self,lr,nbins,alpha,ckpt_path=None,ffcv=False,p_dropout_head=0,**kwargs):
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
        
        aggregation_func="Mean_Aggregation"
        self.classification_head =  Classifier_Head(192,d_hidden=128,t_bins=nbins,p_dropout_head=p_dropout_head)
        self.criterion = Survival_Loss(alpha,ffcv=ffcv)
        self.aggregation = Mean_Aggregation 
        
        
        
    def forward(self, x,y):
        y = torch.rand((x.size(0),0,192)).to(x.device)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        conc_latent = torch.mean(latent,dim=1)

        return mask,latent,ids_restore,conc_latent

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        mask, latent,ids_restore,conc_latent = self(hist_tile,gen)
        pred = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        surv_logits = self.classification_head(conc_latent)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        
        
        loss_Surv = self.criterion(surv_logits,censorship,label)
        
        self.log("train_MAEloss", loss_MAE)
        self.log("train_Survloss",loss_Surv)
        self.log("learning_rate",self.hparams.lr)
        return loss_MAE+loss_Surv
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        mask, latent,ids_restore,conc_latent = self(hist_tile,gen)
        pred = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        surv_logits = self.classification_head(conc_latent)
        
        loss_MAE = self.model.forward_loss(hist_tile, pred, mask)
        loss_Surv = self.criterion(surv_logits,censorship,label)
            
        if stage:
            self.log(f"{stage}mae_loss", loss_MAE,sync_dist=True)
            self.log(f"{stage}surv_loss", loss_Surv,sync_dist=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)
        
    
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,_= batch
        y =  torch.rand((x.size(0),0,192))
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



class Resnet_encoder(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        hist_tile,gen, censorship, label,label_cont = batch
        logits =self(hist_tile)
        return logits
    
    def evaluate(self, batch, stage=None):
        hist_tile,gen, censorship, label,label_cont = batch
        ...
        
    def predict_step(self, batch, batch_idx,mask_ratio=0):
        x,y= batch
        return (self(x))
            
    
    
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

