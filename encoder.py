import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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
    train_loader = RandomModule(train_settings["train_ds_params"]["length"])
    if do_test:
        test_loader = RandomModule(train_settings["test_ds_params"]["length"]) #TODO
    
    #Model
    # TODO model to config settings 
    model = Testnet(**train_settings['model_params'])
    #model.to(rank)

    #Trainer
    trainer = pl.Trainer(
        default_root_dir=default_root_dir, #  TODO
        devices=world_size,
        accelerator="gpu",
        strategy='ddp',
        logger=WandbLogger(save_dir=train_settings["save_dir"], log_model=True) if monitoring else False,
        max_epochs=train_settings['max_epochs']
    )
    
    #training task + testing(optional)
    print(("#"*50+"\n")*2,"Settings:")
    print(train_settings)
    print(("#"*50+"\n")*2,"Start Training!")
    if checkpoint_path is not None:
        print(f"Continue from Checkpointpath: \n {checkpoint_path}")
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader)

    if do_test:
        print(("#"*50+"\n")*2,"Initialize Testing!")
        trainer.test(model, test_loader)# TODO not working yet 
    print(("#"*50+"\n")*2,"Finished Training!")    

def main():
    parser = argparse.ArgumentParser(description="Feature Encoder Training and Encoding")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, required=True, help="train a feature encoder or encode features")
    args = parser.parse_args()

    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        wandb_settings = config["wandb_settings"]
        ...
    # Initialize wandb
    if wandb_settings["monitoring"]:
        wandb.init(project=wandb_settings["project"],
                   entity=wandb_settings["entity"],
                   name=wandb_settings["name"])

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    
    if args.mode=='train':
        # Run training using Slurm's srun
        train_settings = config["train_settings"]
        #mp.spawn(train, args=(num_gpus, train_settings,wandb_settings["monitoring"]), nprocs=num_gpus, join=True)
        train(num_gpus, train_settings,wandb_settings["monitoring"])
    elif args.mode=="encode":
        inference_settings = config["encode_settings"]
        ... # TODO


if __name__ == "__main__":
    main()