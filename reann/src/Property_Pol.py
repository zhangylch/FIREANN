import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmod):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmod

    def forward(self,cart,ef,numatoms,species,atom_index,shifts,create_graph=True):
        pol=jacobian(lambda x: self.get_DM(cart,x,numatoms,species,atom_index,shifts),ef,\
        create_graph=create_graph,vectorize=True)[0]
        return (pol.permute(1,0,2).reshape(-1,9),)

    def get_DM(self,cart,ef,numatoms,species,atom_index,shifts):
        if not ef.requires_grad: ef.requires_grad=True
        species=species.view(-1)
        density = self.density(cart,ef,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output)
        dipole=torch.autograd.grad(varene,ef,\
        create_graph=True,only_inputs=True,allow_unused=True)[0]
        return torch.sum(dipole,dim=0),

