import torch
from torch import nn
from torchvision.models import resnet18

import pytorch_lightning as pl
from torchmetrics import Accuracy
from utils.Aggregation_Utils import Survival_Loss
# Your custom model


class Resnet18Surv(pl.LightningModule):
    def __init__(self,lr,nbins,alpha,tsteps):
        super().__init__()
        self.lr = lr
        self.tsteps = tsteps
        self.save_hyperparameters()
        # Model
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, nbins)
        # Loss
        self.criterion = Survival_Loss(alpha)
    def forward(self, x):
        out = self.resnet18(x)
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
            self.log(f"{stage}_loss", loss, prog_bar=True)
            #self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            )
        
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.tsteps*self.trainer.max_epochs , eta_min=1e-10, last_epoch=- 1, verbose=False)
                ,
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
