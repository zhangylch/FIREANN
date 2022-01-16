import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def Optimize(Epoch,print_epoch,weight_scheduler,scheduler,print_info,data_train,data_test,get_loss,optim): 
     rank=dist.get_rank()
     lr=optim.param_groups[0]["lr"]
     intime_weight=weight_scheduler(lr)
     device=intime_weight.device
     nprop=len(intime_weight)
     for iepoch in range(Epoch): 
         # set the model to train
         get_loss.Prop[0].train()
         loss_train=torch.zeros(nprop,device=device)        
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

         #  print the error of vailadation and test each print_epoch
         if np.mod(iepoch,print_epoch)==0:
             # set the model to eval for used in the model
             # all_reduce the rmse form the training process 
             # here we dont need to recalculate the training error for saving the computation
             dist.reduce(loss_train,0,op=dist.ReduceOp.SUM)
             # calculate the test error
             get_loss.Prop[0].eval()
             loss_test=torch.zeros(nprop,device=device)
             for data in data_test:
                loss=get_loss(data,create_graph=False)
                loss_test+=loss.detach()

             # all_reduce the rmse
             dist.all_reduce(loss_test,op=dist.ReduceOp.SUM)
             if rank==0: print_info(iepoch,lr,loss_train,loss_test)
             loss_scheduler=torch.dot(loss_test,intime_weight)
             # stop criterion before the scheduler
             if lr<=weight_scheduler.end_lr: 
                 print("Normal termination")
                 raise SystemExit
             else:
                 lr=scheduler(lr,loss_scheduler)
                 intime_weight=weight_scheduler(lr)
