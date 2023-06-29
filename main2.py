#!/usr/bin/env python
import wandb
import os
import multiprocessing
import collections
import random

import sys
from utils import  prepare_csv
from models import *
from multi_modal_ds import *
import pandas as pd
import torch
from trainer import *

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_c_index"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    config = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("valid/*", step_metric="epoch")
    
    
    #always same
    storepath = '/work4/seibel/multimodal_survival_prediction/results/gensweep'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "/nodes/bevog/work4/seibel/data/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files" # folderpath for h5y fiels which contain the WSI feat vecs 
    #depends on sweep values
    folds = worker_data.num
    
    batchsize = config["batchsize"]
    bins = config["bins"]
    
    d_gen_out = config["d_gen_out"]
    epochs = 20 # config.epochs #TODO hardcoded to 20 
    learningrate = config["learningrate"]
    alpha = config["alpha"]
    l1_lambda = config["l1_lambda"]
    activation = config["activation"]
    priint(1)
    f = f"/nodes/bevog/work4/seibel/data/tcga_brca_trainable{int(bins)}.csv"
    df = pd.read_csv(f)
    df["kfold"] = df["kfold"].apply(lambda x : (x+1)%folds)

    train_ds = Gen_Dataset(df,data_path = data_path,train=True)
    test_ds = Gen_Dataset(df,data_path = data_path,train=False)
    d_gen = train_ds.gen_depth()
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
    model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device,activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
    criterion = Survival_Loss(alpha)
    
    #training
    
    val_c_index = Gen_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                      test_dataloader,bins,epochs,device,storepath,run_name,
                      l1_lambda 
                      )
    
    run.log(dict(val_c_index=val_c_index))
    wandb.join()
    sweep_q.put(WorkerDoneData(val_c_index=val_c_index))


def main():
    num_folds = 5
    epochs = 20 # hardcoded
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    c_tensor = torch.zeros(size=(num_folds,epochs))
    for num in range(num_folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        c_tensor[num,:] = result.val_c_index
        
    mean_cindex_curve = c_tensor.mean(dim=0)
    if mean_cindex_curve[-1].item() == mean_cindex_curve.max().item():
        last_epoch = True
    else:
        last_epoch = False
    sweep_run.log(dict(val_c_index=mean_cindex_curve.max(),last_epoch=last_epoch))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    main()