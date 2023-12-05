import torch 
from torch import nn 


class Channel_mean(nn.Module):
    def __init__(self):
        super(Channel_mean,self).__init__()
    def forward(self, x):
        return torch.mean(x,dim=1)
    
    
class AttMil(nn.Module):
    """ 
    Implementation of  
    'Attention-based Deep Multiple Instance Learning' 
    by   M. Ilse et al (https://arxiv.org/pdf/1802.04712)
    Gated Attention Mechanism 
    
    Variables: 
    d : dimension of feature vectors
    """
    def __init__(self,d : int):
        super(AttMil,self).__init__()
        self.lin1 = nn.Linear(d,d//2,bias = True)
        torch.nn.init.xavier_normal_(self.lin1.weight)
        self.lin2 = nn.Linear(d//2,d//4,bias=False)
        torch.nn.init.xavier_normal_(self.lin2.weight)
        self.lin3 = nn.Linear(d//2,d//4,bias=False)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        self.lin4 = nn.Linear(d//4,1,bias=False)
        torch.nn.init.xavier_normal_(self.lin4.weight)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.sm = nn.Softmax(dim = 1)

    def forward(self,x):
        x = self.lin1(x)  
        U = self.sigm(self.lin2(x))
        V = self.tanh(self.lin3(x))
        UV= self.lin4(V*U).squeeze(dim=-1) # (B,m,d/4) -> (B,m,1) -> (B,m)
        a = self.sm(UV).unsqueeze(-2)  # (B,m) -> (B,1,m)
        out = torch.bmm(a,x).squeeze(1) # (B,1,m)x(B,m,d/2) ->(B,1,d/2) -> (B,d/2)   or    (m1,m2,...m)x(x1|x2|x3|...)^T  or Σ a_i * x_i
        return out


class SNN(nn.Module):
    """ Implementation of SNN as described in 'Pan-cancer integrative histology-genomic analysis via multimodal deep learning' by R.Chen et al 
    https://pubmed.ncbi.nlm.nih.gov/35944502/ 
    
    Variables:
    d : dimension of molecular vector
    d_out : dimnesion of embedded output vector 
    """
    def __init__(self,d : int,d_out : int = 32,activation="SELU"):
        super(SNN,self).__init__()
        
        
        self.lin1 = nn.Linear(d,256)
        

        self.lin2 = nn.Linear(256,256)
        self.alphadropout1 = nn.AlphaDropout(p=0.5)
        self.alphadropout2 = nn.AlphaDropout(p=0.5)
        self.fc = nn.Linear(256,d_out)
        if activation=="SELU":
            torch.nn.init.normal_(self.lin1.weight, mean=0, std=1/d**0.5)
            torch.nn.init.normal_(self.lin2.weight, mean=0, std=1/256**0.5)
            torch.nn.init.normal_(self.fc.weight, mean=0, std=1/256**0.5)
            self.selu1 = nn.SELU()
            self.selu2 = nn.SELU()
            self.selu3 = nn.SELU()
        elif activation=="RELU":
            torch.nn.init.kaiming_normal_(self.lin1.weight)
            torch.nn.init.kaiming_normal_(self.lin2.weight)
            torch.nn.init.kaiming_normal_(self.fc.weight)
            self.selu1 = nn.ReLU()
            self.selu2 = nn.ReLU()
            self.selu3 = nn.ReLU()
        
        elif activation=="GELU":
            torch.nn.init.kaiming_normal_(self.lin1.weight)
            torch.nn.init.kaiming_normal_(self.lin2.weight)
            torch.nn.init.kaiming_normal_(self.fc.weight)
            self.selu1 = nn.GELU()
            self.selu2 = nn.GELU()
            self.selu3 = nn.GELU()

    def forward(self,x):

        x = self.alphadropout1(self.selu1(self.lin1(x)))
        x = self.alphadropout2(self.selu2(self.lin2(x)))
        x = self.fc(x)
        return self.selu3(x)
    

class Gated_Fusion(nn.Module):
    """ 
    'Pathomic Fusion' by R.Chen et al 
    https://arxiv.org/abs/1912.08937
    Gated Fusion for two modalities 
    
    Variables:
    n_mods : Number of modalities
    dims : list of dimensions for each modality
    modalities : list of trainable feature embeddings from each modality
    """
    def __init__(self,n_mods: int ,dims,device ):
        super(Gated_Fusion,self).__init__()
        self.n_mods = n_mods
        self.dims = dims
        self.device=device
        #H emb
        lin1h = nn.Linear(dims[0],dims[0])
        lin2h = nn.Linear(dims[1],dims[1])
        torch.nn.init.kaiming_normal_(lin1h.weight)
        torch.nn.init.kaiming_normal_(lin1h.weight)
        self.h_emb1 = nn.Sequential(lin1h,nn.ReLU())
        self.h_emb2 = nn.Sequential(lin2h,nn.ReLU())


        #Z emb
        lin_1z = nn.Linear(sum(dims),dims[0])
        lin_2z = nn.Linear(sum(dims),dims[1])
        torch.nn.init.xavier_normal_(lin_1z.weight)
        torch.nn.init.xavier_normal_(lin_2z.weight)
        self.z_emb1 = nn.Sequential(lin_1z,nn.Sigmoid())
        self.z_emb2 = nn.Sequential(lin_2z,nn.Sigmoid())
        """
        self.h_emb_list = nn.ModuleList([])
        self.z_emb_list = nn.ModuleList([])
        for m in range(n_mods):
            h_emb = nn.Sequential(nn.Linear(dims[m],dims[m]),nn.ReLU()) #embeds the m-th  modality into h_m 
            z_emb = nn.Sequential(nn.Linear(sum(dims),dims[m]),nn.Sigmoid()) #m-th embedding of  all modalities in z_m. Must have the same dimensions as h_m for element-wise multiplication.
            self.h_emb_list.append(h_emb)
            self.z_emb_list.append(z_emb)
        """

    def forward(self,mod1,mod2 ):
        #first modality 
        h_1 =  self.h_emb1(mod1) # Projection of first modalitiy 
        z_1 = self.z_emb1(torch.cat((mod1,mod2),dim=1))  # concatenation and projection of all modalities concatenated 
        h_1_gated = h_1*z_1  # Attention
        
        # Same for second modality 
        h_2 =  self.h_emb2(mod2) 
        z_2 = self.z_emb2(torch.cat((mod1,mod2),dim=1))
        h_2_gated = h_2*z_2

        return self.Fusion(h_1_gated,h_2_gated,self.device)
    

    def Fusion(self,v1,v2,device):
        """ 
        Implementation of Ungated Fusion for two modalities as described in 'Pan-cancer integrative histology-genomic analysis via multimodal deep learning' by R.Chen et al 
        https://pubmed.ncbi.nlm.nih.gov/35944502/ 
        No trainable variables, outputs the kronecker product of two vectors. Ones are appended before the kronecker product
        """
        
        B = v1.size(0)
        p = torch.ones((B,1),device=device) 
        v1 = torch.cat((v1,p),dim=-1) # add one values 
        v2 = torch.cat((v2,p),dim=-1) # add one values 
        
        return torch.bmm(v1.unsqueeze(-1),v2.unsqueeze(-2))  

class Classifier_Head(nn.Module):
    def __init__(self,insize,d_hidden=256,t_bins=4,p_dropout_head = 0):
        super(Classifier_Head,self).__init__()
        
        self.linear1 = nn.Linear(insize,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.activ1 = nn.ReLU()
        self.linear2  = nn.Linear(d_hidden,d_hidden)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.activ2 = nn.ReLU()
        self.fc = nn.Linear(d_hidden,t_bins) 
        
        self.dropout1 = nn.Dropout(p=p_dropout_head)
        self.dropout2 = nn.Dropout(p=p_dropout_head)
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.dropout1(self.activ1(self.linear1(x)))
        x = self.dropout2(self.activ2(self.linear2(x)))
        return self.fc(x)
    

class Porpoise(nn.Module):
    """
    Combining all modules to the complete PORPOISE model 
    """
    def __init__(self,d_hist,d_gen,d_gen_out,device,activation,bins,d_hidden=256):
        super(Porpoise,self).__init__()
        dims = [d_hist//2,d_gen_out]   # output dimension of d_hist is given due to structure of implemented AttMIL
        flat_fusion_tensor = (d_hist//2+1) * (d_gen_out+1)
        self.Attn_Mil = AttMil(d=d_hist)
        self.SNN = SNN(d =d_gen ,d_out = d_gen_out,activation=activation)
        self.Gated_Fusion = Gated_Fusion(n_mods = 2,dims=dims,device=device)
        self.Classifier_Head = Classifier_Head(insize = flat_fusion_tensor,d_hidden=d_hidden,t_bins=bins)

    def forward(self,hist,gen):

        hist = self.Attn_Mil(hist)
        gen = self.SNN(gen)

        out = self.Gated_Fusion(hist,gen)

        return self.Classifier_Head(out)


class PrePorpoise(nn.Module):
    """
    Combining all modules to the complete PrePorpoise model 
    """
    def __init__(self,d_hist,d_gen,d_transformer,dropout,activation,bins,d_hidden=256):
        super(PrePorpoise,self).__init__()
        self.SNN = SNN(d =d_gen ,d_out = d_transformer,activation=activation)
        self.lin_embedder1 = nn.Linear(d_hist,d_transformer)
        self.Encoder = nn.TransformerEncoderLayer(d_transformer,nhead=2,dropout=dropout,activation=nn.GELU(),batch_first=True)
        self.lin_embedder2 = nn.Linear(d_transformer,d_transformer//2)
        self.Classifier_Head = Classifier_Head(insize = d_transformer,d_hidden=d_hidden,t_bins=bins)

    def forward(self,hist,gen):
        gen = self.SNN(gen)
        hist = self.lin_embedder1(hist)
        encoded = self.Encoder(torch.cat((gen.unsqueeze(1),hist),dim=1))
        encoded = self.lin_embedder2(encoded)
        gen_out,hist_out = encoded.split([1,encoded.size(1)-1],1)
        out = torch.cat((gen_out.squeeze(1),torch.mean(hist_out,dim=1)),dim=-1)
    
        return self.Classifier_Head(out)


class AttMil_Survival(nn.Module):
    """
    Unimodel: hist survival model 
    """
    def __init__(self,d_hist,bins,device,d_hidden=256):
        super(AttMil_Survival,self).__init__()
        self.AttMil = AttMil(d=d_hist)
        self.Classifier_Head = Classifier_Head(insize = d_hist//2,d_hidden=d_hidden,t_bins=bins)

    def forward(self,hist):
        hist = self.AttMil(hist)
        return self.Classifier_Head(hist)


class SNN_Survival(nn.Module):
    """
    Unimodel: gen survival model 
    """
    def __init__(self,d_gen,d_gen_out,bins,device,activation,d_hidden=256):
        super(SNN_Survival,self).__init__()
        
        self.SNN = SNN(d =d_gen ,d_out = d_gen_out,activation=activation)
        self.Classifier_Head = Classifier_Head(insize = d_gen_out,d_hidden=d_hidden,t_bins=bins)

    def forward(self,gen):
        
        gen = self.SNN(gen)
        return self.Classifier_Head(gen)


###alternativie models
class TransformerMil_Survival(nn.Module):
    """
        Unimodal: hist survival model  with attntion 
    """
    def __init__(self,d_hist,bins,dropout,d_transformer=512, d_hidden = 256):
        super(TransformerMil_Survival,self).__init__()
        d_out = d_hidden 
        self.lin_embedder1 = nn.Linear(d_hist,d_transformer)
        #self.Encoder = torch.nn.TransformerEncoder(nn.TransformerEncoderLayer(d_transformer,
        #                                                                    nhead=8,dropout=0.1,activation=nn.GELU(),batch_first=True)
        #                                        ,num_layers=2)
        self.Encoder = nn.TransformerEncoderLayer(d_transformer,nhead=2,dropout=dropout,activation=nn.GELU(),batch_first=True)
        self.lin_embedder2 = nn.Linear(d_transformer,d_out)
        self.Classifier_Head = Classifier_Head(insize = d_out,d_hidden=d_hidden,t_bins=bins)
    
    def forward(self,x):
        x = self.lin_embedder1(x)
        x = self.Encoder(x)
        out = x.mean(dim=-2)
        out = self.lin_embedder2(out)
        return self.Classifier_Head(out)
    
    
    
class PrePorpoise_meanagg(nn.Module):
    """
    Combining all modules to the complete PORPOISE model 
    """
    def __init__(self,d_hist,d_gen,d_transformer,dropout,activation,bins,attmil,d_hidden=256):
        super(PrePorpoise_meanagg,self).__init__()
        self.SNN = SNN(d =d_gen ,d_out = d_transformer,activation=activation)
        self.lin_embedder1 = nn.Linear(d_hist,d_transformer)
        if attmil:
            self.Encoder = AttMil(d=d_transformer)
            self.Classifier_Head = Classifier_Head(insize = d_transformer//2,d_hidden=d_hidden,t_bins=bins)
        else:
            self.Encoder = nn.Sequential(nn.TransformerEncoderLayer(d_transformer,nhead=2,dropout=dropout,activation=nn.GELU(),batch_first=True),
                                         nn.Linear(d_transformer,d_transformer//2),
                                         Channel_mean(),
                                         )
        self.Classifier_Head = Classifier_Head(insize = d_transformer//2,d_hidden=d_hidden,t_bins=bins)

    def forward(self,hist,gen):
        gen = self.SNN(gen)
        hist = self.lin_embedder1(hist)
        multimodal = torch.cat((gen.unsqueeze(1),hist),dim=1)
        encoded = self.Encoder(multimodal)
        return self.Classifier_Head(encoded)

