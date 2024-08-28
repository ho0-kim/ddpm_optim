import torch
#from torch.optim.optimizer import Optimizer, required



def projected_gradient(x,use_pj=True,use_znorm=True,gc_conv_only=False):
    if use_pj:
      if gc_conv_only:
        if len(list(x.size()))>3:
          if use_znorm:
            x = (x - x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))/(x.std(dim = tuple(range(1,len(list(x.size())))), keepdim = True) + 10e-10)
          else:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
          if use_znorm:
            x = (x - x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))/(x.std(dim = tuple(range(1,len(list(x.size())))), keepdim = True) + 10e-10)
          else:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
    return x   