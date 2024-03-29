import os
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn 
from sksurv.metrics import concordance_index_censored,cumulative_dynamic_auc
import wandb
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

class Survival_Loss(nn.Module):
    def __init__(self,alpha,eps = 1e-7):
        super(Survival_Loss, self).__init__()
        self.alpha = alpha
        self.eps= eps
    
    def forward(self,out,c,t):
        """
        'Bias in Cross-Entropy-Based Training of Deep Survival Networks' by S.Zadeh and M.Schmid 
        https://pubmed.ncbi.nlm.nih.gov/32149626/
        Improved negative loglikeliehood loss  
        
        Variables:
        out : torch.FloatTensor  output logits of the model 
        c : torch.BoolTensor wether the patient is censored(c=1) or ucensored(c=0)
        t : torch.IntTensor label/ground truth of the index where the time-to-event is nested
        alpha : float value within [0,1] weighting the Loss of the censored patients 
        """
        assert out.device == c.device
        h = nn.Sigmoid()(out) #   Hazard function 
        S = torch.cumprod(1-h,dim = -1)  # Survival function
        
        t = t.unsqueeze(-1)
        S_bar = torch.cat((torch.ones_like(t,device=t.device),S),dim=-1) # padded survival function to get acess to the previous time window 

        # gathering the probabilty within the bin with the ground truth index for hazard,survival and padded survival 
        S = S.gather(dim=-1,index=t).clamp(min=self.eps)
        h = h.gather(dim=-1,index=t).clamp(min=self.eps)
        S_bar = S_bar.gather(dim=-1,index = t).clamp(min=self.eps)
        
        
        
        #applying log function  
        logS = torch.log(S)
        logS_bar = torch.log(S_bar)
        logh = torch.log(h).to(c.device)

        #masking by censored or uncensored 
        # L_z(h,S) -> h,S_bar only uncensored,
        # while  L_censored(S) only censored 
        
        logh = logh*(1-c)
        logS_bar = logS_bar*(1-c)
        logS = logS*c

        
        L_z = -torch.mean(logh+logS_bar) # -torch.sum(logS_bar) # only uncensored needed! for h and S_bar 
        L_censored = -torch.mean(logS) # only censored S 
        
        
        return L_z + (1-self.alpha)*L_censored

def c_index(risk_all,c_all,l_all): 
    """
    Variables
    risk_all : FloatTensor must be of shape = (N,)  predicted logits of model 
    c_all : IntTensor must be of shape = (N,) 
    l_all IntTensor must be of shape = (N,)

    Outputs the c-index score 
    """
    with torch.no_grad(): 
        notc = (1-c_all).numpy().astype(bool)
        try:
            c_index = concordance_index_censored(notc,l_all.cpu(),risk_all.detach())
            #print(c_index)
            return c_index[0]
        except:
            print("C index problems")
            return torch.nan
        
    
def risk_func(out):
    h = nn.Sigmoid()(out)
    S = torch.cumprod(1-h,dim = -1)
    risk = -S.sum(dim=1)
    return risk


def prepare_csv(df_path,split,n_bins=4,save = True,kfolds=5,frac_train=None,frac_val=None):
    #splits data into k folds stratified by patients,
    #bins survivaltime and , normalizes/transforms gen features into tensor  
    #returns metadata_csv[index,patientname,path2WSI_bag.pth, survival_bin,survival_time, censorship, k_fold_cluster,], feature tensor
    #fracs (eg:0.8)
    df = pd.read_csv(df_path,compression='zip')
    
    # get time bins 
    df_uncensored = (df[df["censorship"]==0]).drop_duplicates(["case_id"])
    _,bins = pd.qcut(df_uncensored['survival_months'],q = n_bins,retbins=True)  # distribute censored survival months into quartiles
    
    # adapt time bins 
    bins[0] = 0 
    bins[-1] = np.inf
    # bin name = index 
    labels = [i for i in range(n_bins)]
    df.insert(6,"survival_months_discretized",  pd.cut(df["survival_months"],
                                                               bins=bins, 
                                                               labels=labels)) # insert binned survival month
    
    
    if split=="kfold":
        diction = dict([(name,idx) for idx,name in enumerate(df["case_id"].unique()) ])
        df.insert(3,"kfold",df["case_id"].map(diction)%kfolds) # insert kfold 
        genomics = df[df.keys()[11:]]

        scaler = StandardScaler()
        scaled_genomics = scaler.fit_transform(genomics)
        df[df.keys()[11:]] = scaled_genomics

        if save:
            savename = df_path.replace("all_clean.csv.zip",f"_{n_bins}bins_{kfolds}fold.csv")
            df.to_csv(savename,index=False)
        return df 

    
    elif split=="traintestval":
        
        cases = len(df["case_id"].unique())
        train_len = int(cases*frac_train)
        val_len = int(cases*frac_val)
        test_len = cases-train_len-val_len
        
        diction = dict([(name,0) if idx<train_len else (name,1) if idx<train_len+test_len  else (name,2) for idx,name in enumerate(df["case_id"].unique())])
        df.insert(3,"traintest",df["case_id"].map(diction)) # insert traintestlabel 
        
        genomics = df[df.keys()[11:]]
        scaler = StandardScaler()
        scaled_genomics = scaler.fit_transform(genomics)
        df[df.keys()[11:]] = scaled_genomics
        
        df_train = df[df["traintest"]==0]
        df_test = df[df["traintest"]==1]
        df_val = df[df["traintest"]==2]
        if save:
            savename_train = df_path.replace("all_clean.csv.zip",f"_{n_bins}bins_trainsplit.csv")
            df_train.to_csv(savename_train,index=False)
            
            savename_test = df_path.replace("all_clean.csv.zip",f"_{n_bins}bins_testsplit.csv")
            df_test.to_csv(savename_test,index=False)
            
            savename_val = df_path.replace("all_clean.csv.zip",f"_{n_bins}bins_valsplit.csv")
            df_val.to_csv(savename_val,index=False)
        return df_train,df_test,df_val


def KM_wandb(run,risk,c,event_cont,risk_group=None,nbins = 30):
    print("Start Logging KM-Estimators"+ (", with risk_group" if risk_group is not None else  "") )
    
    
    #thresholds
    min,max,mean,median = risk.min(),risk.max(),risk.mean(),risk.median()
    
    #hist
    censored = c.type(torch.bool)
    uncensored = ~c.type(torch.bool)
    
    ###wandb histogram
    x = np.linspace(min,max,nbins)
    x_c1 = torch.histc(risk[censored],bins=nbins,min = min , max =max ).numpy()
    x_c2 = torch.histc(risk[uncensored],bins=nbins,min = min , max =max ).numpy() 

    table_hist = wandb.Table(
            data = do_table(x,x_c1,"censored")+do_table(x,x_c2,"uncensored"),
            columns=["risk", "count","category"],
            )

    fields_hist = {"x":"risk","y":"count","groupKeys":"category","title":"Risk Distribution"}
    custom_histogram = wandb.plot_table(vega_spec_name="tobias-seibel/risk_distribution",
                data_table=table_hist,
                fields = fields_hist )
    
    wandb.log({"Risk Distribution":custom_histogram})
    ###
    
    
    #KaplanMeier Plots
    xfull, yfull = kaplan_meier_estimator(uncensored.numpy(), event_cont)
    
    for thresholds_name,threshold in zip(["mean","median"],[mean,median]): 
        stratification = risk_group if risk_group is not None else risk>=threshold
        
        if (sum(stratification)==0) or sum(~stratification)==0:
            continue
        #Log Rank test 
        group_indicator = (stratification).numpy().astype("bool")
        y_stat = np.asarray([(c,time) for c,time in zip(uncensored,event_cont)],dtype=[('name', '?'), ('field', '<i8')])
        chisq,pvalue = compare_survival(y_stat, group_indicator, return_stats=False)
        
        
        #KM low
        xlow, ylow = kaplan_meier_estimator(uncensored[~stratification].numpy(),
                                    event_cont[~stratification])
        #KM high
        xhigh, yhigh = kaplan_meier_estimator(uncensored[stratification].numpy(),
                                    event_cont[stratification])
        
        
        
        table_KM = wandb.Table(data = do_table(xlow,ylow,f"low risk group {sum(~stratification)}")+do_table(xhigh,yhigh,f"high risk group {sum(stratification)}")+do_table(xfull,yfull,f"total group {len(risk)}"),
                        columns=["time","Survival Probability","Group"],)

        field_KM = {"x":"time","y":"Survival Probability","groupKeys":"Group"}
        custom_KM = wandb.plot_table(vega_spec_name="tobias-seibel/kaplanmeier",
                    data_table=table_KM,
                    fields = field_KM, 
                    string_fields={"title":f"KM RS at {thresholds_name}({round(threshold.item(),2)}),Chi squared = {chisq:.2e}, pvalue = {pvalue:.2e}"},
                    )
        
        run.log({f"KM_{thresholds_name}" :custom_KM,f"{thresholds_name}_pvalue":pvalue,f"{thresholds_name}_statistic":chisq})
    
    wandb.log({"risk_min":min,"risk_max":max,"risk_mean":mean,"risk_median":median})
    print("Finished logging KM-Estimators")
    

def stepfunc(x,y,eps=1e-4):
    #not needed anymore 
    x = np.stack((x-eps,x),axis=1).flatten()
    y = np.stack((y,y),axis=1).flatten()
    return x[1:],y[:-1]

def do_table(x,y,label):
    return [[x[i],y[i],label] for i in range(len(x))]



def dropmissing(df,name,feature_path):
    len_df = len(df)
    df = df.drop(df[df["slide_id"].apply(lambda x : not os.path.exists(os.path.join(feature_path,x.replace(".svs",".h5"))))].index)
    if len_df !=len(df):
        print(f"Dropped {len_df-len(df)}rows in {name} dataframe")
    return df