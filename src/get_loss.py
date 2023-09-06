import numpy as np
import torch
from torch import nn

class Get_Loss():
    def __init__(self,index_prop,Prop_class,Prop_Pol):
        self.loss_fn=nn.MSELoss(reduction="sum")
        self.index_prop=index_prop
        self.Prop=(Prop_class,)
        if Prop_Pol is not None:
            self.Prop+=(Prop_Pol,)

    def __call__(self,data,create_graph=True):
        return self.forward(data,create_graph=create_graph)
 
    def forward(self,data,create_graph=True):
        abProp,cart,ef,numatoms,species,atom_index,shifts=data
        varProp=()
        for iprop in self.Prop:
            varProp+=iprop(cart,ef,numatoms,species,atom_index,shifts,create_graph=create_graph)
        loss=torch.empty(len(self.index_prop),dtype=cart.dtype,device=cart.device)
        # calculate the loss 
        for i,iprop in enumerate(abProp):
            #print(varProp[self.index_prop[i]],iprop,flush=True)
            loss[i]=self.loss_fn(varProp[self.index_prop[i]],iprop)
        return loss
