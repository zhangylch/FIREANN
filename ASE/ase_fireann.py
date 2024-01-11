# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen


modified by Yaolong Zhang for a better efficiency
"""

import torch
import ase.io.vasp
from ase import Atoms
from ase.calculators.fireann import FIREANN
import getneigh as getneigh
from ase.io import extxyz
import time
import numpy as np

ef=np.zeros(3)
fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,100))
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
atomtype = ['O','H']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
maxneigh=25000  # maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
calc=FIREANN(atomtype,maxneigh, ef, getneigh, potential = "PES.pt", device=device, dtype = torch.float32)
start=time.time()
num=0.0
for atoms in configuration:
    calc.reset()
    atoms.calc=calc
    ene = atoms.get_potential_energy(apply_constraint=False)
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    dipole = atoms.get_dipole_moment()
    #print(ene,forces,stress,dipole)
    num+=ene
print(num)
end=time.time()
print(end-start)
