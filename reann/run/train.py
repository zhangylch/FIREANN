#! /usr/bin/env python3
from src.read import *
from src.dataloader import *
from src.optimize import *
from src.density import *
from src.MODEL import *
from src.get_loss import *
from src.scheduler import *
from src.restart import *
from src.weight_scheduler import *
from src.checkpoint import *
from src.save_pes import *
from torch.optim.swa_utils import AveragedModel

if activate=='Tanh_like':
    from src.activate import Tanh_like as actfun
else:
    from src.activate import Relu_like as actfun

if oc_activate=='Tanh_like':
    from src.activate import Tanh_like as oc_actfun
else:
    from src.activate import Relu_like as oc_actfun

#choose the right class used for the calculation of property
if  "Energy" in Prop_list and "Force" not in Prop_list and "Dipole" not in Prop_list:
    from src.Property_E import *
elif "Force" in Prop_list and "Dipole" in Prop_list: 
    from src.Property_E_F_DM import *
elif "Force" in Prop_list and "Dipole" not in Prop_list: 
    from src.Property_E_F import *
elif "Force" not in Prop_list and "Dipole" in Prop_list: 
    from src.Property_E_DM import *
    
if "POL" in Prop_list and len(Prop_list)==1:
    from src.Property_Pol import *
    Property_Pol=None
elif "POL" in Prop_list: 
    import src.Property_Pol as Property_Pol
else:
    Property_Pol=None

from src.cpu_gpu import *

from src.script_PES import *
import pes.PES as PES
if oc_loop==0:
    import lammps.PES as Lammps_PES
else:
    import lammps_REANN.PES as Lammps_PES
from src.print_info import *

#==============================train data loader===================================
dataloader_train=DataLoader(com_coor_train,ef_train,abprop_train,numatoms_train,\
species_train,atom_index_train,shifts_train,batchsize_train,min_data_len=min_data_len,shuffle=True)
#=================================test data loader=================================
dataloader_test=DataLoader(com_coor_test,ef_test,abprop_test,numatoms_test,\
species_test,atom_index_test,shifts_test,batchsize_test,shuffle=False)
# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    data_train=CudaDataLoader(dataloader_train,device,queue_size=queue_size)
    data_test=CudaDataLoader(dataloader_test,device,queue_size=queue_size)
else:
    data_train=dataloader_train
    data_test=dataloader_test

#==============================oc nn module=================================
# outputneuron=nwave for each orbital have a different coefficients
ocmod_list=[]
for ioc_loop in range(oc_loop):
    ocmod=NNMod(maxnumtype,nwave,atomtype,oc_nblock,list(oc_nl),oc_dropout_p,oc_actfun,table_norm=oc_table_norm)
    ocmod_list.append(ocmod)
#=======================density======================================================
getdensity=GetDensity(rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list)
#==============================nn module=================================
nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,initpot=initpot,table_norm=table_norm)
#=========================create the module=========================================
print_info=Print_Info(fout,end_lr,train_nele,test_nele,Prop_list)
Prop_class=Property(getdensity,nnmod).to(device)  # to device must be included
if Property_Pol is not None:
    Prop_Pol=Property_Pol.Property(getdensity,nnmod).to(device)
else:
    Prop_Pol=None


# define the SWA model only on rank 0 and before DDP
swa_model = AveragedModel(Prop_class)

##  used for syncbn to synchronizate the mean and variabce of bn 
#Prop_class=torch.nn.SyncBatchNorm.convert_sync_batchnorm(Prop_class).to(device)
if world_size>1:
    if torch.cuda.is_available():
        Prop_class = DDP(Prop_class, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    else:
        Prop_class = DDP(Prop_class, find_unused_parameters=find_unused)

#initial the class used for evaluating the loss
get_loss=Get_Loss(index_prop,Prop_class,Prop_Pol)

#define optimizer
optim=torch.optim.AdamW(Prop_class.parameters(), lr=start_lr, weight_decay=re_ceff)

# learning rate scheduler 
lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)

# define the class tho save the model for evalutaion
jit_pes=script_pes(PES.PES(),"PES")
jit_lammps=script_pes(Lammps_PES.PES(),"LAMMPS")

# save the checkpoint
checkpoint=Checkpoint(optim)
save_pes=Save_Pes(jit_pes,jit_lammps)

# define the restart class
restart=Restart(optim)

# the scheduler 
scheduler=Scheduler(lr_average,decay_factor,checkpoint,lr_scheduler,restart,optim,Prop_class,swa_model,save_pes)

#scheduler the weight of various properties
weight_scheduler=Weight_Scheduler(init_weight,final_weight,start_lr,end_lr)

# load the model from EANN.pth
if table_init==1: 
    restart(Prop_class,"REANN.pth")
    restart(swa_model,"SWA_REANN.pth")
    nnmod.initpot[0]=initpot
    if rank==0: swa_model.module.nnmod.initpot[0]=initpot
    if optim.param_groups[0]["lr"]>start_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
    if optim.param_groups[0]["lr"]<end_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate
else:
    if rank==0: checkpoint(swa_model,"SWA_REANN.pth")
#for name, m in Prop_class.named_parameters():
#    print(name)
#==========================================================
Optimize(Epoch,print_epoch,weight_scheduler,scheduler,print_info,data_train,data_test,get_loss,optim)
