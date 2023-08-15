import torch 
from torch import nn 
from sksurv.metrics import concordance_index_censored

def c_index(logits_all,c_all,l_all):
    """
    Variables
    logits_all : FloatTensor must be of shape = (N,4)  predicted logits of model 
    c_all : IntTensor must be of shape = (N,) 
    l_all IntTensor must be of shape = (N,)

    Outputs the c-index score 
    """
    
        
    h = nn.Sigmoid()(torch.cat(logits_all,dim=0))
    S = torch.cumprod(1-h,dim = -1)
    risk = -S.sum(dim=1) 
    notc = (1-torch.cat(c_all,dim=0)).cpu().numpy().astype(bool)
    try:
        output = concordance_index_censored(notc, torch.cat(l_all,dim=0).cpu(),risk.cpu())
        c_inidex = output[0]
    except:
        print("WARNING: C-INDEX ISSUE ,probably all samples are censored, return NAN")
        c_index = float('nan')
    return c_index