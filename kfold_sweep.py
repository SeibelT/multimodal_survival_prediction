# Framework for k-fold crossvalidation wandb sweeps adapted from https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation
#
import os
import wandb
import torch
import collections
import pandas as pd
import multiprocessing
import argparse
from models.Aggregation_Models import *
from trainer.Aggregation_Trainer import *
from datasets.Aggregation_DS import *
from utils.Aggregation_Utils import Survival_Loss,c_index,KM_wandb

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config","n_folds")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ["val_c_all","risk_fold","c_fold","l_con_fold"])


def dropmissing(df,name,feature_path):
        len_df = len(df)
        df = df.drop(df[df["slide_id"].apply(lambda x : not os.path.exists(os.path.join(feature_path,x.replace(".svs",".h5"))))].index)
        if len_df !=len(df):
            print(f"Dropped {len_df-len(df)}rows in {name} dataframe")
        return df
    
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
    #get worker data 
    fold = worker_data.num
    num_fold = worker_data.n_folds
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #get config data from sweep file
    batchsize = config["batchsize"]
    bins = config["bins"]
    d_gen_out = config["d_gen_out"]
    epochs = config["epochs"] 
    learningrate = config["learningrate"]
    alpha = config["alpha"]
    l1_lambda = config["l1_lambda"]
    activation = config["activation"]
    modality = config["modality"]
    assert modality in ["Porpoise","PrePorpoise","gen","hist","hist_attention"],"Modality name not known"
    dropout = config["dropout"]
    datapath = config["datapath"] #absolute path  '"/work4/seibel/data'
    csv_path = config["csv_path"]
    d_hist,feature_path = config["dim_hist_and_feature_path"]
    gen_augmentation = config["gen_augmentation"]
    #setup file paths and read CSV #TODO more general solution needed if time 
    storepath = os.path.join(datapath,f"/results/{modality}sweep") # not used! 
    
    csv_path = os.path.join(csv_path,"tcga_brca_trainable"+str(bins)+".csv") # CSV file 
    
    
    
    df = pd.read_csv(csv_path)
    df = dropmissing(df,"kfold dataframe",feature_path)
    df["kfold"] = df["kfold"].apply(lambda x : (x+fold)%num_fold) 
    
    #Initialize Dataset and Model based on Modality
    if modality=="Porpoise":
        train_ds = HistGen_Dataset(df,data_path = feature_path,train=True,gen_augmentation=gen_augmentation)
        val_ds = HistGen_Dataset(df,data_path = feature_path,train=False,gen_augmentation=gen_augmentation)
        d_gen = train_ds.gen_depth()
        model = Porpoise(d_hist=d_hist,d_gen=d_gen,d_gen_out=d_gen_out,device=device,activation=activation,bins=bins).to(device)
        
    
    elif modality=="PrePorpoise":
        train_ds = HistGen_Dataset(df,data_path = feature_path,train=True,gen_augmentation=gen_augmentation)
        val_ds = HistGen_Dataset(df,data_path = feature_path,train=False,gen_augmentation=gen_augmentation)
        d_gen = train_ds.gen_depth()
        model = PrePorpoise(d_hist=d_hist,d_gen=d_gen,d_transformer=d_hist//4,dropout=dropout,activation=activation,bins=bins).to(device)
        
    
    elif modality=="gen":
        train_ds = Gen_Dataset(df,data_path = feature_path,train=True,gen_augmentation=gen_augmentation)
        val_ds = Gen_Dataset(df,data_path = feature_path,train=False,gen_augmentation=gen_augmentation)
        d_gen = train_ds.gen_depth()
        model = SNN_Survival(d_gen,d_gen_out,bins=bins,device=device,activation=activation).to(device)
        
    
    elif modality=="hist":
        train_ds = Hist_Dataset(df,data_path = feature_path,train=True)
        val_ds = Hist_Dataset(df,data_path = feature_path,train=False)
        model = AttMil_Survival(d_hist=d_hist,bins=bins,device=device).to(device)
        
    elif modality=="hist_attention":
        train_ds = Hist_Dataset(df,data_path = feature_path,train=True)
        val_ds = Hist_Dataset(df,data_path = feature_path,train=False)
        model = TransformerMil_Survival(d_hist=d_hist,bins=bins,dropout=dropout).to(device)
    
   
    criterion = Survival_Loss(alpha) 
    training_dataloader = torch.utils.data.DataLoader( train_ds,batch_size=batchsize)
    val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=batchsize)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate,betas=[0.9,0.999],weight_decay=1e-5,)
    
    #run trainer
    if modality in ["Porpoise","PrePorpoise",]:
        c_vals,risk_fold,c_fold,l_con_fold = MM_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality,
                    )
        
    elif modality in ["gen","hist","hist_attention"]:
        c_vals,risk_fold,c_fold,l_con_fold = Uni_Trainer_sweep(run,model,optimizer,criterion,training_dataloader,
                    val_dataloader,bins,epochs,device,storepath,run_name,
                    l1_lambda,modality=modality
                    )
   
    
    run.log(dict(val_c_all=c_vals.numpy()))
    wandb.join()
    
    sweep_q.put(WorkerDoneData(val_c_all=c_vals.numpy(),risk_fold=risk_fold.to(torch.float16).numpy(),c_fold=c_fold.to(torch.int16).numpy(),l_con_fold=l_con_fold.to(torch.float16).numpy()))
    


def main(num_folds):
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

    metrics,risk_all,c_all,l_con_all = [],[],[],[]
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
        c_all.append(torch.tensor(result.c_fold))
        l_con_all.append(torch.tensor(result.l_con_fold))
        risk_all.append(torch.tensor(result.risk_fold,dtype=torch.float64))
        
    metrics_mean = np.mean(np.asarray(metrics),axis=0)
    KM_total = KM_wandb(sweep_run,torch.cat(risk_all),torch.cat(c_all),torch.cat(l_con_all))
    
    sweep_run.log(dict(c_index_max=metrics_mean.max(),c_index_last=metrics_mean[-1],c_index_epoch=np.argmax(metrics_mean),KM_total=KM_total))
    wandb.join()
    print("riskallsize",torch.cat(risk_all).size())
    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wandb sweep with k-fold cross validation")
    parser.add_argument("--folds", type=int, required=False,default=4, help="Number of folds")
    
    args = parser.parse_args()
    num_folds = args.folds
    main(num_folds)