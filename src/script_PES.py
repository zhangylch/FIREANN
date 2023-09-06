import torch
from collections import OrderedDict

class script_pes():
    def __init__(self,pes,name):
        self.pes=pes
        self.name=name

    def __call__(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict['reannparam'].items():
            if k[0:7]=="module.":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        self.pes.load_state_dict(new_state_dict)
        scripted_pes=torch.jit.script(self.pes)
        for params in scripted_pes.parameters():
            params.requires_grad=False
        scripted_pes.to(torch.double)
        scripted_pes.save(self.name+".pt")
