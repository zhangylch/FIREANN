import torch
import numpy as np
import torch.distributed as dist

# define the strategy of weight decay
class Scheduler():
    def __init__(self,lr_average,decay_factor,checkpoint,lr_scheduler,restart,optim,model,swa_model,save_pes):
        self.best_loss = 1e30
        self.rank=dist.get_rank()
        self.lr_average=lr_average
        self.decay_factor=decay_factor
        self.checkpoint=checkpoint
        self.lr_scheduler=lr_scheduler
        self.restart=restart 
        self.optim=optim
        self.swa_model=swa_model
        self.model=model
        self.save_pes=save_pes
    
    def __call__(self,lr,loss):
        return self.forward(lr,loss)
 
    def forward(self,lr,loss):
        if loss>25.0*self.best_loss or loss.isnan():
            self.restart(self.model,"REANN.pth")
            self.restart(self.swa_model,"SWA_REANN.pth")
            self.optim.param_groups[0]["lr"]=self.optim.param_groups[0]["lr"]*self.decay_factor
        else:
            if self.rank==0:
                if lr<=self.lr_average: 
                    self.swa_model.update_parameters(self.model)
                    self.save_pes(self.swa_model.module)
                else:
                    self.save_pes(self.model)
            
            if loss<self.best_loss:
                self.best_loss=loss.item()
                if self.rank==0:
                    self.checkpoint(self.swa_model,"SWA_REANN.pth")
                    self.checkpoint(self.model,"REANN.pth")
                 
        self.lr_scheduler.step(loss)
        return self.optim.param_groups[0]["lr"]
