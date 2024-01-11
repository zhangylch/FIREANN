#----------------fireann interface is for FIREANN package-------------------------


import numpy as np
import os
import torch
import re
#from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)

class FIREANN(Calculator):

    implemented_properties = ['energy', 'forces', 'stress', 'dipole','energies']

    def __init__(self, atomtype, maxneigh, ef, getneigh, nn = 'PES.pt',device="cpu",dtype=torch.float32,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.atomtype = atomtype
        self.maxneigh=maxneigh
        self.getneigh=getneigh
        pes=torch.jit.load(nn)
        pes.to(self.device).to(self.dtype)
        pes.eval()
        self.cutoff=pes.cutoff
        self.pes=torch.jit.optimize_for_inference(pes)
        self.ef=torch.from_numpy(ef).to(self.device).to(self.dtype).unsqueeze(0)
        self.tcell=[]
        #self.pes=torch.compile(pes)
    
    def calculate(self,atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        cell=np.array(self.atoms.cell)
        if "cell" in system_changes:
            if cell.ndim==1:
                cell=np.diag(cell)
            self.getneigh.init_neigh(self.cutoff,self.cutoff/2.0,cell.T)
            self.tcell=torch.from_numpy(cell).to(self.dtype).to(self.device).unsqueeze(0)
        icart = self.atoms.get_positions()
        cart,neighlist,shiftimage,scutnum=self.getneigh.get_neigh(icart.T,self.maxneigh)
        cart=torch.from_numpy(cart.T).contiguous().to(self.device).to(self.dtype).unsqueeze(0)
        neighlist=torch.from_numpy(neighlist[:,:scutnum]).contiguous().to(self.device).to(torch.long)
        shifts=torch.from_numpy(shiftimage.T[:scutnum,:]).contiguous().to(self.device).to(self.dtype)
        symbols = list(self.atoms.symbols)
        species = [self.atomtype.index(i) for i in symbols]
        species = torch.tensor(species,device=self.device,dtype=torch.long)
        if "dipole" in properties:
            self.ef.requires_grad=True
        else:
            self.ef.requires_grad=False

        if "forces" in properties:
            cart.requires_grad=True
        else:
            cart.requires_grad=False

        if "stress" in properties:
            self.tcell.requires_grad=True
        else:
            self.tcell.requires_grad=False
        atomic_ene=self.pes(self.tcell,cart,self.ef,torch.zeros(scutnum,device=self.device,dtype=torch.long),neighlist,shifts,species)
        if "energies" in properties: self.results['energies'] = atomic_ene.detach().numpy()
        energy = torch.sum(atomic_ene)
        self.results['energy'] = float(energy.detach().numpy())
        if "forces" in properties and "stress" in properties:
            forces,stress = -torch.autograd.grad(energy,[cart,self.tcell])
            forces = forces.squeeze(0).detach().numpy()
            self.results['forces'] = forces
            stress = stress.squeeze(0).detach().numpy()
            self.results['stress'] = stress

        if "forces" in properties and "stress" not in properties:
            forces =-torch.autograd.grad(energy,cart)[0].squeeze(0)
            forces = forces.detach().numpy()
            self.results['forces'] = forces

        if "stress" in properties and "forces" not in properties:
            stress = -torch.autograd.grad(energy,self.tcell)[0].squeeze(0)
            stress = stress.detach().numpy()
            self.results['stress'] = stress

        if "dipole" in properties:
            dipole=torch.autograd.grad(energy,self.ef)[0].squeeze(0)
            dipole = dipole.detach().numpy()
            self.results['dipole'] = dipole
