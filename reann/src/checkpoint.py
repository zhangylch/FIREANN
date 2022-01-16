import torch

class Checkpoint():
    def __init__(self,optim):
        self.optim=optim
    
    def __call__(self,model,checkfile):
        state = {'reannparam': model.state_dict(), 'optimizer': self.optim.state_dict()}
        torch.save(state, checkfile)

