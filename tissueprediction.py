import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torchmetrics import Accuracy

import wandb
import os

import random
from models.Encoder_Models import *
from datasets.Tile_DS import *
from utils.Encoder_Utils import create_feature_ds

class VitTinyTissue(pl.LightningModule):
    def __init__(self,lr,nbins,ckpt_path=None,):
        super().__init__()
        self.lr = lr
        self.nbins = nbins
        self.mask_ratio = 0
        self.ids_shuffle = None
        self.save_hyperparameters()
        #ViT 
        self.model = mae_vit_tiny_patch16()
        #load weights
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = {k.replace("module.model.", ""): v for k, v in ckpt["model"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        
        
        self.classification_head = nn.Linear(192,nbins) 
        self.acc = Accuracy(task="multiclass", num_classes=9)
        self.criterion = nn.CrossEntropyLoss() 
        
        
    def forward(self, x):
        y = torch.rand((x.size(0),0,192),device=x.device)
        latent, mask, ids_restore, ids_shuffle = self.model.forward_encoder(x, self.mask_ratio,y, self.ids_shuffle)
        latent_mean = torch.mean(latent,dim=1)
        logits = self.classification_head(latent_mean)
        return logits

    def training_step(self, batch, batch_idx):
        hist_tile,labels = batch
        logits = self(hist_tile)
        
        loss = self.criterion(logits,labels)
        acc = self.acc(logits, labels)
        
        self.log("train_loss", loss)
        self.log(f"train_acc", acc, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        hist_tile,labels = batch
        logits = self(hist_tile)
        acc = self.acc(logits, labels)
        loss = self.criterion(logits,labels)    
        if stage:
            self.log(f"{stage}loss", loss,sync_dist=False)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        
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
    
    
    
class TissueDataset(Dataset):
    def __init__(self,f,transform,mode,rand=False,frac=0.8): # TODO
        super(TissueDataset,self).__init__()
        
        
        filelist = []
        for root, dirs, files in os.walk(f, topdown=False):
            for name in files:
                if name.endswith("tif"):
                    filelist.append(os.path.join(root, name) )
                
        filelist.sort()
        if rand:
            random.Random(66).shuffle(filelist)
        
        labellist = [file.split("/")[-2] for file in filelist]

        if mode =="train": 
            filelist = filelist[:int(len(filelist)*frac)]
            labellist = labellist[:int(len(labellist)*frac)]
        elif mode =="val":
            filelist = filelist[int(len(filelist)*frac):]
            labellist = labellist[int(len(labellist)*frac):]

        mapper = {'ADI':0, 'BACK':1, 'DEB':2, 'LYM':3, 'MUC':4, 'MUS':5, 'NORM':6, 'STR':7, 'TUM':8}
        
        self.filelist = filelist
        self.labellist = list(map(mapper.get, labellist))

        
        self.transforms = transform
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self,idx):
        
        tile = Image.open(self.filelist[idx])
        tile = self.transforms(tile)
        
        label = torch.tensor(self.labellist[idx])
        #label_onehot =torch.zeros((9,))
        #label_onehot[self.labellist[idx]] = 1
        return (tile,label)
        
class TileModule(pl.LightningDataModule):
    def __init__(self,frac,rand,batch_size,num_workers,**kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size 
        self.frac = frac
        self.rand = rand
        self.ftrain = "/nodes/bevog/work4/seibel/data/tissueDS/NCT-CRC-HE-100K"
        self.fval = self.ftrain
        self.ftest = "/nodes/bevog/work4/seibel/data/tissueDS/CRC-VAL-HE-7K"
        self.transform_train =  transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5)
                                                    ]
                                                   )
        self.transform_eval =  transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                                    ]
                                                   )
        
    def setup(self, stage):
        self.train_set = TissueDataset(f = self.ftrain,transform = self.transform_train,mode="train",rand=self.rand,frac=self.frac)
        self.test_set = TissueDataset(f = self.fval,transform = self.transform_eval,mode="val",rand=self.rand,frac=self.frac)
        self.val_set = TissueDataset(f = self.ftest,transform=self.transform_eval,mode=None)
        
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=False)


name = "run3"
monitoring = True
logging = True
frac=0.8
rand=True
batch_size= 128
num_workers=0
lr = 3e-4
nbins = 9
max_epochs = 30
max_steps = -1
ckpt_path = "/nodes/bevog/work4/seibel/data/mae_tiny_400e.pth.tar"
log_every_n_steps = 1
do_test = False
checkpoint_path = None
slurm = True if "SLURM_JOB_ID" in os.environ else False 
if monitoring :
    if slurm:       
        if  (int(os.environ['SLURM_PROCID'])==0):   
            wandb.init(project="TissueClassifier",
                        entity="tobias-seibel",
                        name=name,
                        save_code = True,
                        )
    else:
        wandb.init(project="TissueClassifier",
                        entity="tobias-seibel",
                        name=name,
                        save_code = True,
                        )
        

data_module = TileModule(frac,rand,batch_size,num_workers)
model =   VitTinyTissue(lr,nbins,ckpt_path=ckpt_path)
if monitoring and logging:
    wandb_logger = WandbLogger(log_model="all") 
    wandb_logger.watch(model,log_freq=100*log_every_n_steps,log="all")

trainer = pl.Trainer(
            default_root_dir = "/home/seibel/tissueclassifier", #  TODO
            devices = 1,
            accelerator = "gpu",
            log_every_n_steps=log_every_n_steps,
            max_steps = max_steps,
            logger = wandb_logger if monitoring else False,
            max_epochs = max_epochs,
            callbacks = [StochasticWeightAveraging(swa_lrs=1e-2,annealing_strategy="cos")],
                            )



print(("#"*50+"\n")*2,"Start Training!")
if checkpoint_path is not None:
    print(f"Continue from Checkpointpath: \n {checkpoint_path}")
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
else:
        trainer.fit(model, data_module)
        
if do_test:
    print(("#"*50+"\n")*2,"Initialize Testing!")
    trainer.test(model, data_module) 
print(("#"*50+"\n")*2,"Finished Training!")    

# finish wandb 
if monitoring:
    wandb.finish()