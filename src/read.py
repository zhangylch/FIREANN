import os
import gc
import time
import torch
import numpy as np
from src.read_data import *
from src.get_info_of_rank import *
from src.gpu_sel import *
# used for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# open a file for output information in iterations
fout=open('nn.err','w')
# save the dimension of each prop
Prop_dict={"Energy":1,"Dipole":3,"POL":9}
# the whole list of properties
Prop_full_list=["Energy",["Dipole","Force"],"POL"]

# global parameters for input_nn
Prop_list_init={"Energy":0.1,"Dipole":0.1,"Force":50,"POL":0.1}
Prop_list_final={"Energy":0.1,"Dipole":0.1,"Force":0.5,"POL":0.1}
table_coor=0                   # 0: cartestion coordinates used 1: fraction coordinates used
table_init=0                   # 1: a pretrained or restart  
nblock = 1                     # the number of resduial NN blocks
ratio=0.9                      # ratio for vaildation
#==========================================================
Epoch=50000                  # total numbers of epochs for fitting 
patience_epoch=100              # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
decay_factor=0.5               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
print_epoch=10                 # number of epoch to calculate and print the error
# adam parameter                 
start_lr=0.01                  # initial learning rate
end_lr=1e-4                    # final learning rate
#==========================================================
# regularization coefficence
re_ceff=0.0                 # L2 normalization cofficient
batchsize_train=32                  # batch size 
batchsize_val=512                  # batch size 
nl=[128,128]                  # NN structure
dropout_p=[0.0,0.0]           # dropout probability for each hidden layer
activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
queue_size=10
table_norm = False
find_unused = False
#===========param for orbital coefficient ===============================================
oc_loop = 0
oc_nl = [32,32]          # neural network architecture   
oc_nblock = 1
oc_dropout_p=[0.0,0.0,0.0,0.0]
#=====================act fun===========================
oc_activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
#========================queue_size sequence for laod data into gpu
oc_table_norm=False
DDP_backend="nccl"
# folder to save the data
folder="./"
dtype='float32'   #float32/float64
singma=3               # init singma for GTO
norbit=None
#=====================ema==================
ema_decay=0.99
ema_nbatch=32
#=========================================

#=====================environment for select the GPU in free===============================
local_rank = int(os.environ.get("LOCAL_RANK"))
local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

# select gpu
if local_rank==0: os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_info')
#======================read input_nn=================================================================
with open('para/input_nn','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32

# define the number of the neuron of output layer
outputneuron=1

# set the default type as double
torch.set_default_dtype(torch_dtype)

#======================read input_density=============================================
# defalut values in input_density
nipsin=2
cutoff=4.0
nwave=6
with open('para/input_density','r') as f1:
    while True:
        tmp=f1.readline()
        if not tmp: break
        string=tmp.strip()
        if len(string)!=0:
            if string[0]=='#':
               pass
            else:
               m=string.split('#')
               exec(m[0])

# increase the nipsin
nipsin+=1

#========================use for read rs/inta or generate rs/inta================
maxnumtype=len(atomtype)
if 'rs' in locals().keys():
    rs=torch.from_numpy(np.array(rs,dtype=np_dtype))
    inta=torch.from_numpy(np.array(inta,dtype=np_dtype))
    nwave=rs.shape[1]
else:
    rs=torch.rand(maxnumtype,nwave)*cutoff
    inta=-torch.ones_like(rs)/(singma*singma)

if not norbit:
    norbit=int((nwave+1)*nwave/2*(nipsin))
nl.insert(0,norbit)
oc_nl.insert(0,norbit)

#=============================================================================
folder_train=folder+"train/"
folder_val=folder+"validation/"
# obtain the number of system
folderlist=[folder_train,folder_val]
# read the configurations and physical properties
# to oredr the Prop_list and the corresponding weight
Prop_list=[]
init_weight=[]
final_weight=[]
for key in Prop_list_init.keys():
    Prop_list.append(key)
    init_weight.append(Prop_list_init[key])
    final_weight.append(Prop_list_final[key])

start_table=None
if "Force" in Prop_list_init.keys(): start_table=1

numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,ef,abprop,force=Read_data(folderlist,Prop_list,start_table=start_table)

# to validate the number of properties equal to the number of read from datafile and set the corresponding weight
nprop=0
#Prop_list.remove("Force")
for m in Prop_list:
    if m != "Force": nprop+=Prop_dict[m]

nprop1=0
for i in range(len(abprop)):
    nprop1+=len(abprop[i][0])
if nprop!=nprop1:
    raise RuntimeError("The length of Property is not consistent with the dimension of properties defined in 'input_nn'. Please check the input or configuration file")

#============================convert form the list to torch.tensor=========================
numpoint=np.array(numpoint,dtype=np.int64)
numatoms=np.array(numatoms,dtype=np.int64)
# here the double is used to scal the potential with a high accuracy
# here to convert the abprop to torch tensor
for i,iprop in enumerate(abprop):
    abprop[i]=np.array(iprop)
    
initpot=0.0
if "Energy" in Prop_list:
    initpot=np.sum(abprop[0][:])/np.sum(numatoms)
    abprop[0]=abprop[0]-initpot*numatoms.reshape(-1,1)
# get the total number configuration for train/val
ntotpoint=0
for ipoint in numpoint:
    ntotpoint+=ipoint

#define golbal var
if numpoint[1]==0: 
    numpoint[0]=int(ntotpoint*ratio)
    numpoint[1]=ntotpoint-numpoint[0]

# parallel process the variable  
gpu_sel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank)
#a=torch.empty(10000,device=device)  # used for apply some memory to prevent two process on the smae gpu
world_size = int(os.environ.get("WORLD_SIZE"))
dist.init_process_group(backend=DDP_backend)

if batchsize_train<world_size or batchsize_val<world_size:
    raise RuntimeError("The batchsize used for training or val dataset are smaller than the number of processes, please decrease the number of processes.")
# device the batchsize to each rank
batchsize_train=int(batchsize_train/world_size)
batchsize_val=int(batchsize_val/world_size)
#=======get the minimal data in each process for fixing the bug of different step for each process
min_data_len_train=numpoint[0]-int(np.ceil(numpoint[0]/world_size))*(world_size-1)
min_data_len_val=numpoint[1]-int(np.ceil(numpoint[1]/world_size))*(world_size-1)
if min_data_len_train<=0 or min_data_len_val<=0:
    raise RuntimeError("The size of training or validation dataset are smaller than the number of processes, please decrease the number of processes.")
# devide the work on each rank
# get the shifts and atom_index of each neighbor for train
rank=dist.get_rank()
rank_begin=int(np.ceil(numpoint[0]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[0]/world_size))*(rank+1),numpoint[0])
range_train=[rank_begin,rank_end]
com_coor_train,force_train,numatoms_train,species_train,atom_index_train,shifts_train=\
get_info_of_rank(range_train,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

# get the shifts and atom_index of each neighbor for validation
rank_begin=int(np.ceil(numpoint[1]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[1]/world_size))*(rank+1),numpoint[1])
range_val=[numpoint[0]+rank_begin,numpoint[0]+rank_end]
com_coor_val,force_val,numatoms_val,species_val,atom_index_val,shifts_val=\
get_info_of_rank(range_val,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_val,cutoff,device,np_dtype)

abprop_train=()
abprop_val=()
train_nele=[]
val_nele=[]
tmp_f=0
for i,iprop in enumerate(Prop_list):
    if iprop == "Force":
        abprop_train+=(force_train,)
        abprop_val+=(force_val,)
        train_nele.append(np.sum(numatoms[0:numpoint[0]])*3)
        val_nele.append(np.sum(numatoms[numpoint[0]:ntotpoint])*3)
        tmp_f=1
    else:
        abprop_train+=(torch.from_numpy(abprop[i-tmp_f][range_train[0]:range_train[1]]).to(torch_dtype),)
        abprop_val+=(torch.from_numpy(abprop[i-tmp_f][range_val[0]:range_val[1]]).to(torch_dtype),)
        train_nele.append(numpoint[0]*Prop_dict[Prop_list[i]])
        val_nele.append(numpoint[1]*Prop_dict[Prop_list[i]])
ef_train=torch.from_numpy(np.array(ef[range_train[0]:range_train[1]],dtype=np_dtype))
ef_val=torch.from_numpy(np.array(ef[range_val[0]:range_val[1]],dtype=np_dtype))
train_nele=torch.from_numpy(np.array(train_nele)).to(device)
val_nele=torch.from_numpy(np.array(val_nele)).to(device)
# delete the original coordiante
del coor,mass,numatoms,atom,scalmatrix,period_table
if start_table==0: del abprop
if start_table==1: del abprop,force,ef
gc.collect()
    
#======================================================
patience_epoch=patience_epoch/print_epoch
init_weight=torch.from_numpy(np.array(init_weight,dtype=np_dtype)).to(device)
final_weight=torch.from_numpy(np.array(final_weight,dtype=np_dtype)).to(device)

# dropout_p for each hidden layer
dropout_p=np.array(dropout_p,dtype=np_dtype)
oc_dropout_p=np.array(oc_dropout_p,dtype=np_dtype)
#==========================================================
if dist.get_rank()==0:
    fout.write("REANN Package used for fitting energy and tensorial Property\n")
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()

# used for obtaining the index of each property in Property calculation 
# the architecture is scalar, first derivate, sencond derivate
# only incorporate the property with reference into the loss function
num=0
index_prop=[]
for iprop in Prop_full_list:
    if isinstance(iprop,str):
        for i in Prop_list:
           if i==iprop: 
              index_prop.append(num)
    else:
        num-=1
        for iiprop in iprop:
            for i in Prop_list:
                if i==iiprop:
                    num+=1
                    index_prop.append(num)
    num+=1
# only fitting the polarizability
if "POL" in Prop_list and len(Prop_list)==1:
    index_prop=[0]
