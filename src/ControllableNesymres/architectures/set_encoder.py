import torch
import torch.nn as nn
import pytorch_lightning as pl
from .set_transformer import ISAB, PMA
from .utils import float2bit

class SetEncoder(pl.LightningModule):
    def __init__(self,cfg):
        super(SetEncoder, self).__init__()
        self.linear = cfg.linear
        self.bit16 = cfg.bit16
        self.norm = cfg.norm
        assert cfg.linear != cfg.bit16, "one and only one between linear and bit16 must be true at the same time" 
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))
            
        self.activation = cfg.activation
        self.input_normalization = cfg.input_normalization
        if cfg.linear:
            self.linearl = nn.Linear(cfg.dim_input,16*cfg.dim_input)
        self.selfatt = nn.ModuleList()
        #dim_input = 16*dim_input
        self.selfatt1 = ISAB(16*cfg.dim_input, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln)
        for i in range(cfg.n_l_enc):
            self.selfatt.append(ISAB(cfg.dim_hidden, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln))
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads, cfg.num_features, ln=cfg.ln)


   
    
    def forward(self, x):
        
        if self.bit16:
            x = float2bit(x, device=self.device)
            x = x.view(x.shape[0],x.shape[1],-1)
            if self.norm:
                x = (x-0.5)*2
        if self.input_normalization:
            means = x[:,:,-1].mean(axis=1).reshape(-1,1)
            std = x[:,:,-1].std(axis=1).reshape(-1,1)
            std[std==0] = 1
            x[:,:,-1] = (x[:,:,-1] - means)/std
            
        if self.linear:
            if self.activation == 'relu':
                x = torch.relu(self.linearl(x))
            elif self.activation == 'sine':
                x = torch.sin(self.linearl(x))
            else:
                x = (self.linearl(x))
        x = self.set_encoder_forward(x)
        return x

    def set_encoder_forward(self, x):
        # print(x.shape)
        # breakpoint()
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        #print(x.shape)
        return x




class SymEncoder(pl.LightningModule):
    def __init__(self,cfg):
        super(SymEncoder, self).__init__()
        self.embedding = nn.Embedding(25, cfg.dim_hidden)
        self.linear = nn.Linear(25,cfg.num_features)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.linear(x)
        return x.permute(0,2,1)


if __name__ == "__main__":
    model = SetEncoder(2,2,6,2,3,1,3,1,'linear',0,0,1,1,True)
    #self, n_l,dim_input,dim_hidden,num_heads,num_inds,ln,num_features,linear,activation,bit16,norm,mean,std,input_normalization
    print(model)
    model.eval()
    x = torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]]).T.unsqueeze(0).float()
    x1 = torch.Tensor([[6,3,2,4,5,1],[12,9,8,10,11,7]]).T.unsqueeze(0).float()
    print(x.max())
    print(model(x))
    print(model(x1))