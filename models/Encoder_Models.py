import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

# Your custom model


class Testnet(pl.LightningModule):
    def __init__(self,channels,lr):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = nn.Sequential(nn.Conv2d(3, channels, kernel_size=(3, 3), padding='same', bias=True),
                          nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(1),
                          nn.Linear(channels,1),
                          nn.Flatten(0))
        self.Acc = Accuracy(task="binary")

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits.float(),y.float())
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits.float(),y.float())
        preds = nn.Sigmoid()(logits)
        acc = self.Acc(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // 64 # fALSE!!!
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps_per_epoch*self.trainer.max_epochs , eta_min=0, last_epoch=- 1, verbose=False)
                ,
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
