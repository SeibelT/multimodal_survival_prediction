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
    
    do_test = train_settings["do_test"]  
    default_root_dir = train_settings["default_root_dir"]  
    checkpoint_path = train_settings["checkpoint_path"] 
    #Data
    data_module = TileModule(**train_settings["dataset_params"])
    #data_module.setup("fit")
    #training_length = data_module.train_set.__len__()
    #batch_size = train_settings["dataset_params"]["batch_size"]
    #T_steps = training_length //batch_size 
    
    #Model
    model =  SupViTSurv(**train_settings["model_params"])
    
    #wandb monitoring + watching weights
    if monitoring:
        wandb_logger = WandbLogger(save_dir=train_settings["save_dir"], log_model="all") 
        wandb_logger.watch(model)
    
    

    #Trainer
    trainer = pl.Trainer(
        default_root_dir=default_root_dir, #  TODO
        devices=-1,
        accelerator="gpu",
        num_nodes = 1,
        max_steps=10,
        profiler="simple",
        strategy= "auto" if world_size==1 else "ddp",
        logger=wandb_logger if monitoring else False,
        max_epochs=train_settings["max_epochs"],
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2,annealing_strategy="cos",)],
                        )
    
    if False:  # run with one gpu only to find ideal values for bs and lr -> adapt to multigpu 
        tuner =Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=data_module, mode="binsearch",) #not supported for ddp 8pl version 2.0.6 -> run on single single gpu 
        lr_finder = tuner.lr_find(model, datamodule=data_module)
        print(lr_finder.results)
        print(lr_finder.suggestion())
        fig = lr_finder.plot(suggest=True)
        fig.savefig("./lrplot.png",dpi='figure',format="png")
    
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


    # finish wandb 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Encoder Training and Encoding")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, required=True, help="train a feature encoder or encode features")
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as file:
    
        config = yaml.safe_load(file)
        wandb_settings = config["wandb_settings"]
    
    
    # Initialize wandb
    if wandb_settings["monitoring"] and (int(os.environ['SLURM_PROCID'])==0) :
        print("Initialize WANDB on ",int(os.environ['SLURM_PROCID']))
        wandb.init(project=wandb_settings["project"],
                   entity=wandb_settings["entity"],
                   name=wandb_settings["name"],
                   config = config,
                   save_code = True,
                   )

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"World Size:",num_gpus)
    
    
    
    mode = args.mode
    if mode=='train':
        # Run training using Slurm's srun
        train_settings = config["train_settings"]
        train(num_gpus, train_settings,wandb_settings["monitoring"])
    elif mode=="encode":
        inference_settings = config["encode_settings"]
        ... # TODO
    
    