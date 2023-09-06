import torch

# define the strategy of weight decay
class Save_Pes():
    def __init__(self,PES_Normal,PES_Lammps):
        self.PES=PES_Normal
        self.Lammps=PES_Lammps
    
    def __call__(self,model):
        state = {'reannparam': model.state_dict()}
        self.PES(state)
        if self.Lammps:
            self.Lammps(state)

