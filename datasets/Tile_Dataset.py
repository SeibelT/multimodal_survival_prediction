import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import pytorch_lightning as pl


class RandomModule(pl.LightningDataModule):
  def __init__(self, length):
      super().__init__()
      self.train_set = Random_Dataset(8*length)
      self.val_set = Random_Dataset(4*length)
        
  def setup(self, stage):
    # transforms for images
    transform= transforms.Compose([])
    
      
    

  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=64)

  def val_dataloader(self):
    return DataLoader(self.val_set, batch_size=64)



class Random_Dataset(Dataset):
    def __init__(self,length=20):
        self.length = length
        self.label = torch.randint(0,2,size=(length,))
        
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        y = self.label[idx]
        if y.item():
            x = torch.rand(3,32,32)
        else:
            x = torch.normal(0.5,0.5,size=(3,32,32))
        return x,y