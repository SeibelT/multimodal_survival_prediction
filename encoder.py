import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging
import wandb
import os

from torchvision import datasets, transforms

from models.Encoder_Models import *
from datasets.Tile_Dataset import *


def train( world_size, train_settings,monitoring):
    #torch.cuda.set_device(rank)
    checkpoint_path = train_settings["checkpoint_path"]  
    do_test = train_settings["do_test"]  
    default_root_dir = train_settings["default_root_dir"]  
    
    #Data
    data_module = TileModule(**train_settings["dataset_params"])
    data_module.setup("fit")
    training_length = data_module.train_set.__len__()
    batch_size = train_settings["dataset_params"]["batch_size"]
    T_steps = training_length //batch_size 
    
    #Model
    model = Resnet18Surv(**train_settings['model_params'],tsteps=T_steps)
    #wandb.watch(model)

    #Trainer
    trainer = pl.Trainer(
        default_root_dir=default_root_dir, #  TODO
        devices=-1,
        accelerator="gpu",
        num_nodes = 1,
        strategy="ddp",
        logger=WandbLogger(save_dir=train_settings["save_dir"], log_model=True) if monitoring else False,
        max_epochs=train_settings['max_epochs'],
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        
                        )
    #tuner =Tuner(trainer)
    #tuner.scale_batch_size(model, datamodule=data_module, mode="power") not supported for ddp 8pl version 2.0.6
    #tuner.lr_find(model)
    
    #training task + testing(optional)
    print(("#"*50+"\n")*2,"Settings:")
    print(train_settings)
    print(("#"*50+"\n")*2,"Start Training!")
    if checkpoint_path is not None:
        print(f"Continue from Checkpointpath: \n {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)

    if do_test:
        print(("#"*50+"\n")*2,"Initialize Testing!")
        trainer.test(model, data_module) # TODO not working yet 
    print(("#"*50+"\n")*2,"Finished Training!")    

def main(conf):
    #parser = argparse.ArgumentParser(description="Feature Encoder Training and Encoding")
    #parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    #parser.add_argument("--mode", type=str, required=True, help="train a feature encoder or encode features")
    #args = parser.parse_args()

    
    with open(conf, 'r') as file:
        config = yaml.safe_load(file)
        wandb_settings = config["wandb_settings"]
    mode = config["mode"]
    
    # Initialize wandb
    if wandb_settings["monitoring"] and (os.environ['SLURM_PROCID']==0):
        print("Initialize WANDB")
        wandb.init(project=wandb_settings["project"],
                   entity=wandb_settings["entity"],
                   name=wandb_settings["name"],
                   config = config,
                   save_code = True,
                   )

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"World Size:",num_gpus)
    
    if mode=='train':
        # Run training using Slurm's srun
        train_settings = config["train_settings"]
        #mp.spawn(train, args=(num_gpus, train_settings,wandb_settings["monitoring"]), nprocs=num_gpus, join=True)
        train(num_gpus, train_settings,wandb_settings["monitoring"])
    elif mode=="encode":
        inference_settings = config["encode_settings"]
        ... # TODO


if __name__ == "__main__":
    main("./encoder_configs/base.yaml")
    