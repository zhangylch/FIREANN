import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

# define the strategy of weight decay
class Save_Scheduler():
    def __init__(self,init_weight,final_weight,start_lr,end_lr,scheduler,optim,Prop_class,PES_Normal,PES_Lammps):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.best_loss = 1e8
        self.rank=dist.get_rank()
        self.model=Prop_class
        self.scheduler=scheduler
        self.optim=optim
        self.PES=PES_Normal
        self.Lammps=PES_Lammps
    
    def __call__(self,loss):
        return self.forward(loss)
 
    def forward(self,loss):
        if loss<self.best_loss:
             if self.rank == 0:
                 state = {'eannparam': self.model.state_dict(), 'optimizer': self.optim.state_dict()}
                 torch.save(state, "./EANN.pth")
                 self.best_loss=loss
                 self.PES.jit_pes()
                 if self.Lammps:
                     self.Lammps.jit_pes()
        self.scheduler.step(loss)
        lr=self.optim.param_groups[0]["lr"]
        intime_weight=self.init_weight+(self.final_weight-self.init_weight)*(lr-self.start_lr)*(self.end_lr-self.start_lr+1e-8)
        return lr,intime_weight
