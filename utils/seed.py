import numpy as np
import random
import torch

def setup_seed(params):
     torch.manual_seed(params.seed) #cpu
     torch.cuda.manual_seed_all(params.seed) #gpu
     np.random.seed(params.seed) #numpy
     random.seed(params.seed)
     torch.backends.cudnn.deterministic = True #cudnn
     torch.backends.cudnn.benchmark = True