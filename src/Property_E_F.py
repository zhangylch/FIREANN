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
        cart.requires_grad=True
        species=species.view(-1)
        density = self.density(cart,ef,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        force = -torch.autograd.grad(torch.sum(varene),cart,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0]
        return (varene.view(-1,1),force.view(numatoms.shape[0],-1))

