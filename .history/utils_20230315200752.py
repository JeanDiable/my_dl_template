import os
import shutil
import torch
import yaml
import random
import numpy as np
import time,datetime
from torchvision import transforms

def load_config() ->dict :
    with open("./config.yml") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

def copy_code(outdir):
    """copy files to the outdir to store the complete script with each experiment as a log"""
    code = []
    exclude = set([])
    for root,_,files in os.walk(".",topdpwn = True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]
    for r,f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.makedir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """a class for storing values and update values"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
    
def save_tensor_to_png(img : torch.Tensor, path : str):
    toPIL= transforms.ToPILImage()
    pic = toPIL(img)
    pic.save(path)






