import torch
from get_neigh import *
from torch.autograd.functional import jacobian,hessian

class Calculator():
    def __init__(self,device,torch_dtype):
        #load the serilizable model
        pes=torch.jit.load("PES.pt")
        # FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
        pes.to(device).to(torch_dtype)
        # set the eval mode
        pes.eval()
        self.cutoff=pes.cutoff
        self.pes=torch.jit.optimize_for_inference(pes)

    def get_dipole(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy)
        dipole=torch.autograd.grad(varene,ef,create_graph=create_graph)[0]
        return dipole,

    def get_ene_dipole(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        dipole=torch.autograd.grad(torch.sum(varene),ef,create_graph=create_graph)[0]
        return varene,dipole,
         
    def get_ene(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        return varene,
    
    def get_pol(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy)
        dipole=torch.sum(torch.autograd.grad(varene,ef,create_graph=True)[0],dim=0)
        pol=torch.cat([torch.autograd.grad(idipole,ef,create_graph=True)[0].view(-1,1,3) for idipole in dipole],dim=1)
        return pol,

    def get_stresstensor(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        stress=torch.autograd.grad(torch.sum(varene),cell,create_graph=create_graph)[0]
        return stress

    def get_ene_stress(self,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        stress=-torch.autograd.grad(torch.sum(varene),cell,create_graph=create_graph)[0]
        return varene,stress
#===============bug need to be fixed in jit for vectorize=======================================     
    #def get_pol(self,cart,ef,neigh_list,shifts,species,create_graph=False):
    #    pol=jacobian(lambda x: self.get_dipole_for_pol(cart,x,neigh_list,shifts,species),ef,\
    #    create_graph=create_graph,vectorize=True)[0].view(3,-1,3).permute(1,0,2)
    #    return (pol,)
    #
    #def get_dipole_for_pol(self,cart,ef,neigh_list,shifts,species):
    #    atomic_energy=self.pes(cart,ef,neigh_list,shifts,species)
    #    varene=torch.sum(atomic_energy)
    #    dipole=torch.autograd.grad(varene,ef,create_graph=True)[0]
    #    return torch.sum(dipole,dim=0),
#=======================================================================================
