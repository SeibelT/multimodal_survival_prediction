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
