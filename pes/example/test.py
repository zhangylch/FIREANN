import numpy as np
import torch
from gpu_sel import *
from calculate import *
from write_format import *
import getneigh as getneigh
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
cutoff=calculator.cutoff
maxneigh=25000  # maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
# same as the atomtype in the file input_density
atomtype=["H","O"]
# save the lattic parameters
num=0
nbatch=2
batchsize=8
with open("configuration",'r') as f1:
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
    cell=np.array(cell)
    cart=np.array(cart)
    species=torch.from_numpy(np.array(species)).to(device)  
    pbc=torch.from_numpy(np.array(pbc)).to(device).to(torch.long).view(-1,3)
    ef=torch.from_numpy(np.array(ef)).to(device).to(torch_dtype).view(-1,3)
    # neigh list 
    num=0
    totnum=ef.shape[0]
    print(totnum)
    for m in range(nbatch):
        neigh_list=torch.empty(2,0).to(device).to(torch.long)
        shifts=torch.empty(0,3).to(device).to(torch_dtype)
        index_cell=ti=torch.empty(0).to(device).to(torch.long)
        num_up=min(num+batchsize,totnum)
        bcart=cart[num:num_up]
        bpbc=pbc[num:num_up]
        bcell=cell[num:num_up]
        bspecies=species[num:num_up]
        bef=ef[num:num_up]
        c_cart=[]
        for i in range(num_up-num):
            getneigh.init_neigh(cutoff,cutoff/2.0,bcell[i].T)
            coor,neighlist,shiftimage,scutnum=getneigh.get_neigh(bcart[i].T,maxneigh)
            index_cell=torch.cat((index_cell,torch.ones(scutnum).to(device).to(torch.long)*i),0)
            c_cart.append(coor.T)
            tmp_neigh=neighlist+i*bcart.shape[1]
            neigh_list=torch.cat((neigh_list,torch.from_numpy(tmp_neigh)[:,:scutnum].to(device).to(torch.long)),1)
            shifts=torch.cat((shifts,torch.from_numpy(shiftimage).T[:scutnum].to(device).to(torch_dtype)),0)
            num+=1
        
        bcart=torch.from_numpy(np.array(c_cart)).contiguous().to(device).to(torch_dtype)
        bcell=torch.from_numpy(bcell).to(device).to(torch_dtype)
        bef.requires_grad=False
        bcell.requires_grad=True
        bcart.requires_grad=False
        varene,stress=calculator.get_ene_stress(bcell,bcart,bef,index_cell,neigh_list,shifts,bspecies.view(-1))
        varene=varene.detach().cpu().numpy()
        stress=stress.detach().cpu().numpy()
        bcell.requires_grad=False
        bcell[:,1,1]=bcell[:,1,1]-1e-4
        varene1,=calculator.get_ene(bcell,bcart,bef,index_cell,neigh_list,shifts,bspecies.view(-1))
        bcell[:,1,1]=bcell[:,1,1]+2e-4
        varene2,=calculator.get_ene(bcell,bcart,bef,index_cell,neigh_list,shifts,bspecies.view(-1))
        stress1=-(varene2-varene1)/2e-4
        print("hello",stress[:,1,1],stress1)
