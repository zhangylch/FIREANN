import numpy as np
import torch
from gpu_sel import *
from calculate import *
from write_format import *
import re
# unit
length_a_au=1/5.29177208590000E-01
Di_de_au=0.393430307
energy_ev_au=1.0/2.72113838565563E+01
Di_de_ev= 0.20819434
Ef_au_ev=51.4221
#convertor
factor_energy=energy_ev_au
factor_dipole=Di_de_au/Di_de_ev
factor_pol=Di_de_au/Di_de_ev*Ef_au_ev
convertor=np.array([factor_energy,factor_dipole,factor_pol])
t_convertor=1.0/convertor
#====================================Prop_list================================================
Prop_list=["Energy","Dipole","POL"]
pattern=[]
for prop in Prop_list:
    if prop!="Force":
        pattern.append(re.compile(r"(?<={}=)\'(.+?)\'".format(prop)))

# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype
torch_dtype=torch.double
calculator=Calculator(device,torch_dtype)
# same as the atomtype in the file input_density
atomtype=["C","O","N","H"]
# save the lattic parameters
num=0
nbatch=20
batchsize=10
with open("/group/zyl/data/NMA/b3lyp-ef/configuration",'r') as f1:
    species=[]
    cart=[]
    cell=[]
    pbc=[]
    ef=[]
    abprop=[]
    while True:
        string=f1.readline()
        if not string: break
        species.append([])
        cart.append([])
        cell.append([])
        pbc.append([])
        ef.append([])
        string=f1.readline()
        cell[num].append(list(map(float,string.split())))
        string=f1.readline()
        cell[num].append(list(map(float,string.split())))
        string=f1.readline()
        cell[num].append(list(map(float,string.split())))
        string=f1.readline()
        pbc[num].append(list(map(float,string.split()[1:4])))
        string=f1.readline()
        abprop.append([])
        for i,ipattern in enumerate(pattern):
            tmp=re.findall(ipattern,string)
            abprop[num].append(np.array(list(map(float,tmp[0].split()))))

        while True:
            string=f1.readline()
            if "External_field:" in string: 
                ef[num].append(list(map(float,string.split()[1:4])))
                break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart[num].append(tmp1[0:3])
            species[num].append(atomtype.index(tmp[0]))
        num+=1
    species=torch.from_numpy(np.array(species)).to(device)  
    pbc=torch.from_numpy(np.array(pbc)).to(device).to(torch.long).view(-1,3)
    cart=torch.from_numpy(np.array(cart)).to(device).to(torch_dtype)  
    cell=torch.from_numpy(np.array(cell)).to(device).to(torch_dtype)
    ef=torch.from_numpy(np.array(ef)).to(device).to(torch_dtype).view(-1,3)
    # neigh list 
    num=0
    totnum=ef.shape[0]
    for m in range(nbatch):
        neigh_list=torch.empty(2,0).to(device).to(torch.long)
        shifts=torch.empty(0,3).to(device).to(torch_dtype)
        num_up=min(num+batchsize,totnum)
        bcart=cart[num:num_up]
        bpbc=pbc[num:num_up]
        bcell=cell[num:num_up]
        bspecies=species[num:num_up]
        bef=ef[num:num_up]
        for i in range(num_up-num):
            holder=calculator.get_neighlist(bpbc[i],bcart[i],bcell[i])
            tmp_neigh=holder[0]+i*cart.shape[1]
            neigh_list=torch.cat((neigh_list,tmp_neigh),1)
            shifts=torch.cat((shifts,holder[1]),0)
            num+=1
        
        bef.requires_grad=True
        bcart.requires_grad=False
        varene,dipole=calculator.get_ene_dipole(bcart,bef,neigh_list,shifts,\
        bspecies.view(-1))
        pol=calculator.get_pol(bcart,bef,neigh_list,shifts,bspecies.view(-1))[0]
        varene=varene.detach().cpu().numpy()
        dipole=dipole.detach().cpu().numpy()
        print(pol)
        pol=pol.view(-1,9).detach().cpu().numpy()
        init_num=num-bef.shape[0]
        for i in range(bef.shape[0]):
            print(abprop[init_num+i][0],varene[i])
            print(abprop[init_num+i][1],dipole[i])
            print(abprop[init_num+i][2],pol[i])
