import torch
import numpy as np
import torch.distributed as dist

# define the strategy of weight decay
class Scheduler():
    def __init__(self,end_lr,decay_factor,checkpoint,lr_scheduler,restart,optim,model,swa_model,save_pes):
        self.best_loss = 1e30
        self.rank=dist.get_rank()
        self.end_lr=end_lr
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
            dist.barrier()
            self.restart(self.model,"REANN.pth")
            self.restart(self.swa_model,"SWA_REANN.pth")
            self.optim.param_groups[0]["lr"]=self.optim.param_groups[0]["lr"]*self.decay_factor
        else:
            # store the best loss for preventing a boomm of error
            if loss<self.best_loss:
                self.best_loss=loss.item()
                if self.rank==0:
                    # begin to update the SWA model
                    self.save_pes(self.model)
                    # store the checkpoint at each epoch
                    self.checkpoint(self.swa_model,"SWA_REANN.pth")
                    self.checkpoint(self.model,"REANN.pth")

                 
        self.lr_scheduler.step(loss)
        return self.optim.param_groups[0]["lr"]
