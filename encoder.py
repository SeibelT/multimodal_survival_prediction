import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging,ModelCheckpoint
import wandb
import os

from models.Encoder_Models import *
from datasets.Tile_DS import *
from utils.Encoder_Utils import create_feature_ds

def train(world_size, train_settings, monitoring):
    assert os.path.exists(train_settings["save_dir"]), "save_dir does not exist!"
    assert os.path.exists(train_settings["default_root_dir"]),"save_dir does not exist!"
    
    do_test = train_settings["do_test"]  
    default_root_dir = train_settings["default_root_dir"]  
    checkpoint_path = train_settings["checkpoint_path"] 
    #Data
    if train_settings["ffcv"]:
        from datasets.Tile_DS_ffcv import ffcvmodule
        data_module = ffcvmodule(**train_settings["dataset_params"],is_dist=True if world_size>1 else False)
    else:
        data_module = globals()[train_settings["datamodule"]](**train_settings["dataset_params"])
    print("Datamodule :",data_module.__class__.__name__)
    

    #Model
    model =  globals()[train_settings["model_name"]](ffcv = train_settings["ffcv"],**train_settings["model_params"])
    if checkpoint_path is not None:
        print(f"Load model from Checkpointpath: \n {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    
    print("Modelname :",model._get_name())
    
    callbacks = [#ModelCheckpoint(dirpath=default_root_dir,every_n_epochs=train_settings["max_epochs"]//4 if train_settings["max_epochs"]>10 else 1 ,save_top_k=-1)],
                ModelCheckpoint(dirpath=default_root_dir,every_n_epochs=25  ,save_top_k=-1),]
    callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2,annealing_strategy="cos",
                                               annealing_epochs=train_settings["annealing_epochs"]))  if train_settings["stochastic_weightaveraging"] else print("No StochasticWeightAveraging")
    
    #wandb monitoring + watching weights
    if monitoring:
        wandb_logger = WandbLogger(save_dir=train_settings["save_dir"], log_model="all") 
        if train_settings["monitor_weights_gradients"]:
            wandb_logger.watch(model,log_freq=20*train_settings["log_every_n_steps"],log="all")
    
    #Trainer
    if world_size>1:
        trainer = pl.Trainer(
            default_root_dir = default_root_dir, 
            devices = world_size,
            accelerator = "gpu",
            log_every_n_steps=train_settings["log_every_n_steps"],
            num_nodes = int(os.environ['SLURM_JOB_NUM_NODES']),
            max_steps = train_settings["max_steps"],
            profiler = train_settings["profiler"],
            strategy = "ddp",#"ddp_find_unused_parameters_true"
            logger = wandb_logger if monitoring else False,
            max_epochs = train_settings["max_epochs"],
            callbacks = callbacks,
                            )
    else:
        trainer = pl.Trainer(
        default_root_dir=default_root_dir, 
        accelerator="gpu",
        devices=1,
        log_every_n_steps=train_settings["log_every_n_steps"],
        max_steps=train_settings["max_steps"],
        profiler=train_settings["profiler"],
        logger=wandb_logger if monitoring else False,
        max_epochs=train_settings["max_epochs"],
        callbacks = callbacks,
                           )
    
    if train_settings["tune"]:  # run with one gpu only to find ideal values for bs and lr -> adapt to multigpu 
        tuner =Tuner(trainer)
        #tuner.scale_batch_size(model, datamodule=data_module, mode="binsearch",) #not supported for ddp 8pl version 2.0.6 -> run on single single gpu 
        lr_finder = tuner.lr_find(model, datamodule=data_module)
        print(lr_finder.results)
        print(lr_finder.suggestion())
        fig = lr_finder.plot(suggest=True)
        fig.savefig("./lrplot_survMAE.png",dpi='figure',format="png")
    
    #training
    print(("#"*50+"\n")*2,"Settings:")
    print(train_settings)
    print(("#"*50+"\n")*2,"Start Training!")
    if checkpoint_path is not None:
        print(f"Continue from Checkpointpath: \n {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)
    #testing
    if do_test:
        print(("#"*50+"\n")*2,"Initialize Testing!")
        trainer.test(model, data_module) 
    print(("#"*50+"\n")*2,"Finished Training!")    
    
    # finish wandb 
    if monitoring:
        wandb.finish()

def encode(inference_settings):
    save_path = inference_settings["save_path"] 
    new_ds_name = inference_settings["new_ds_name"] 
    mycheckpnt = inference_settings["mycheckpnt"]  # TODO rename this->?  and ckpt_path-> pretrainedMAEI1K_path or something
    encode_gen = inference_settings["encode_gen"]
    batch_size = inference_settings["batch_size"]
    num_workers = inference_settings["num_workers"]
    pin_memory = inference_settings["pin_memory"]
    model =  globals()[inference_settings["model_name"]](ffcv = False,encode_gen=encode_gen,**inference_settings["model_params"])
    
    if mycheckpnt is not None:
        print("Load Modelweights")
        model.load_state_dict(torch.load(mycheckpnt)["state_dict"])


    transform  = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                            ]
                                        )
    df_tile_slide_path = inference_settings["df_tile_slide_path"]
    df_data_paths = inference_settings["df_data_paths"]
    cntd=inference_settings["cntd"]
    for df_data_path in df_data_paths:
        print("current path",df_data_path)
        create_feature_ds(save_path,new_ds_name,model,transform,df_tile_slide_path,df_data_path,gen=encode_gen,cntd=cntd,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Encoder Training and Encoding")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, required=True, help="train a feature encoder or encode features")
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as file:
    
        config = yaml.safe_load(file)
        wandb_settings = config["wandb_settings"]
    
    #Slurm
    slurm = True if "SLURM_JOB_ID" in os.environ else False 
    
    #Wandb
    if wandb_settings["monitoring"] and slurm and not (args.mode=="encode"):
        if  (int(os.environ['SLURM_PROCID'])==0):
            print("Initialize WANDB on ",int(os.environ['SLURM_PROCID']))
            wandb.init(project=wandb_settings["project"],
                       entity=wandb_settings["entity"],
                       name=wandb_settings["name"],
                       config = config,
                       save_code = True,
                       )
    elif wandb_settings["monitoring"] and not slurm and not (args.mode=="encode"):
        wandb.init(project=wandb_settings["project"],
                   entity=wandb_settings["entity"],
                   name=wandb_settings["name"],
                   config = config,
                   save_code = True,
                   )
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count() if slurm else 1 # multi gpu only on slurm
    print(f"World Size:",num_gpus)
    
    mode = args.mode
    if mode=='train':
        # Run training using Slurm's srun
        train_settings = config["train_settings"]
        train(num_gpus, train_settings,wandb_settings["monitoring"])
    elif mode=="encode":
        inference_settings = config["encode_settings"]
        encode(inference_settings)
    
    