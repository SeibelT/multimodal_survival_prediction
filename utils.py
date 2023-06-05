import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn 



def prepare_csv(df_path,k,n_bins=4,savename = None):
    #clusters data into k folds stratified by patients,
    #bins survivaltime and , normalizes/transforms gen features into tensor  
    #returns metadata_csv[index,patientname,path2WSI_bag.pth, survival_bin,survival_time, censorship, k_fold_cluster,], feature tensor
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
    df.insert(3,"kfold",df.index%k) # insert kfold 

    genomics = df[df.keys()[11:]]

    scaler = StandardScaler()
    scaled_genomics = scaler.fit_transform(genomics)
    df[df.keys()[11:]] = scaled_genomics

    if savename is not None:
        df.to_csv(savename,index=False)
    return df 




def survival_loss(out,c,t,alpha):
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
    
    h = nn.Sigmoid()(out) #   Hazard function 
    S = torch.cumprod(1-h,dim = -1)  # Survival function

    
    S_bar = torch.cat((torch.ones_like(t),S),dim=1) # padded survival function to get acess to the previous time window 

    # gathering the probabilty within the bin with the ground truth index for hazard,survival and padded survival 
    S = S.gather(dim=-1,index=t)
    h = h.gather(dim=-1,index=t)
    S_bar = S_bar.gather(dim=-1,index = t)
    
    
    
    #applying log function  
    logS = torch.log(S)
    logS_bar = torch.log(S_bar)
    logh = torch.log(h)

    #masking by censored or uncensored 
    # L_z(h,S) -> h,S_bar only uncensored,
    # while  L_censored(S) only censored 
    logh = logh.masked_select(~c)
    logS_bar = logS_bar.masked_select(~c)
    logS = logS.masked_select(c)


    L_z = -torch.sum(logh+logS_bar) # -torch.sum(logS_bar) # only uncensored needed! for h and S_bar 
    L_censored = -torch.sum(logS) # only censored S 
    
    
    return L_z + (1-alpha)*L_censored