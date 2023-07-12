#
#
#
# Code adapted from https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation
#
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
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config","n_folds")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_c_all"))


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
    ######
    fold = worker_data.num
    num_fold = worker_data.n_folds
    batchsize = config["batchsize"]
    bins = config["bins"]
    
    d_gen_out = config["d_gen_out"]
    epochs = config["epochs"] # config.epochs #TODO hardcoded to 20 
    learningrate = config["learningrate"]
    alpha = config["alpha"]
    l1_lambda = config["l1_lambda"]
    activation = config["activation"]
    modality = config["modality"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = config["dropout"]
    datapath = config["datapath"] #absolute path  '"/work4/seibel/data'
    
    storepath =    datapath+f"/results/{modality}sweep"
    feature_path = datapath+"/TCGA-BRCA-DX-features/tcga_brca_20x_features/pt_files/"
    f =            datapath+"/tcga_brca_trainable"+str(bins)+".csv"
    
    df = pd.read_csv(f)
    df["kfold"] = df["kfold"].apply(lambda x : (x+fold)%num_fold)
    
    if modality=="gen":
        train_ds = Gen_Dataset(df,data_path = feature_path,train=True)
        test_ds = Gen_Dataset(df,data_path = feature_path,train=False)
        d_gen = train_ds.gen_depth()
        model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device,activation=activation).to(device)
        
    
    elif modality=="hist":
        train_ds = Hist_Dataset(df,data_path = feature_path,train=True)
        test_ds = Hist_Dataset(df,data_path = feature_path,train=False)
        model = AttMil_Survival(d_hist=2048,bins=bins,device=device).to(device)
        
    elif modality=="hist_attention":
        train_ds = Hist_Dataset(df,data_path = feature_path,train=True)
        test_ds = Hist_Dataset(df,data_path = feature_path,train=False)
        model = TransformerMil_Survival(d_hist=2048,bins=bins,dropout=dropout).to(device)
    
   
    criterion = Survival_Loss(alpha) 
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=batchsize)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
    c_vals = Uni_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    test_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,batchsize=batchsize
                    )
    #######
    
    run.log(dict(val_c_all=c_vals.numpy()))
    wandb.join()
    sweep_q.put(WorkerDoneData(val_c_all=c_vals.numpy()))


def main():
    num_folds = 5

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

    metrics = []
    for num in range(num_folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                n_folds=num_folds,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_c_all)
    metrics_mean = np.mean(np.asarray(metrics),axis=0)
    
    sweep_run.log(dict(c_index_max=metrics_mean.max(),c_index_last=metrics_mean[-1],c_index_epoch=np.argmax(metrics_mean)))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    main()