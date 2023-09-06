import numpy as np
import torch 
import time

class Print_Info():
    def __init__(self,fout,end_lr,train_nele,val_nele,Prop_list):
        self.ntrain=train_nele
        self.nval=val_nele
        self.fout=fout
        self.end_lr=end_lr
        # print the required information
        self.fout.write("{:<8}{:<8}{:<6}".format("Epoch","lr","train"))
        for iprop in Prop_list:
            self.fout.write("{:<11}".format(iprop))
        self.fout.write("{:<6}".format("val")) 
        for iprop in Prop_list:
            self.fout.write("{:<11}".format(iprop))
        self.fout.write("\n")
        

    def __call__(self,iepoch,lr,loss_train,loss_val):
        self.forward(iepoch,lr,loss_train,loss_val) 
   
    def forward(self,iepoch,lr,loss_train,loss_val):
        loss_train=torch.sqrt(loss_train/self.ntrain).cpu()
        loss_val=torch.sqrt(loss_val/self.nval).cpu()
        self.fout.write("{:<8}{:<8.1e}{:6}".format(iepoch,lr,"RMSE"))
        for iprop in loss_train:
            self.fout.write("{:<10.3e} ".format(iprop))
        self.fout.write("{:<6}".format("RMSE"))
        for iprop in loss_val:
            self.fout.write("{:<11.3e}".format(iprop))
        self.fout.write("\n")
        self.fout.flush()
        
