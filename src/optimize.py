import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def Optimize(Epoch,print_epoch,ema_nbatch,weight_scheduler,scheduler,print_info,data_train,data_val,get_loss,swa_model,optim): 
     rank=dist.get_rank()
     lr=optim.param_groups[0]["lr"]
     intime_weight=weight_scheduler(lr)
     device=intime_weight.device
     nprop=len(intime_weight)
     for iepoch in range(Epoch): 
         # set the model to train
         get_loss.Prop[0].train()
         loss_train=torch.zeros(nprop,device=device)        
         num=0
         for data in data_train:
             loss=get_loss(data)
             loss_train+=loss.detach()
             loss=torch.dot(loss,intime_weight)
             # clear the gradients of param
             #optim.zero_grad()
             optim.zero_grad(set_to_none=True)
             # print(torch.cuda.memory_allocated)
             # obtain the gradients
             loss.backward()
             optim.step()   
             #update the EMA model
             if np.mod(num,ema_nbatch)==0: swa_model.update_parameters(scheduler.model)
             num+=1

         #  print the error of vailadation and  each print_epoch
         if np.mod(iepoch,print_epoch)==0:
             # set the model to eval for used in the model
             # all_reduce the rmse form the training process 
             # here we dont need to recalculate the training error for saving the computation
             dist.reduce(loss_train,0,op=dist.ReduceOp.SUM)
             # calculate the val error
             get_loss.Prop[0].eval()
             loss_val=torch.zeros(nprop,device=device)
             for data in data_val:
                loss=get_loss(data,create_graph=False)
                loss_val+=loss.detach()

             # all_reduce the rmse
             dist.all_reduce(loss_val.detach(),op=dist.ReduceOp.SUM)
             if rank==0: print_info(iepoch,lr,loss_train,loss_val)
             loss_scheduler=torch.dot(loss_val,intime_weight)
             # stop criterion before the scheduler
             if lr<=scheduler.end_lr:
                 break
             else:
                 lr=scheduler(lr,loss_scheduler)
                 intime_weight=weight_scheduler(lr)
