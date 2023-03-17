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

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class LogProcessBar():
    def __init__(self,logfile,args,print_log):
        self.last_time = time.time()
        self.begin_time = time.time()
        self.logfile = logfile
        self.print_log = print_log

        if not os.path.exists(os.path.dirname(logfile)):
            os.makedirs(os.path.dirname(logfile),exist_ok = True)
        with open(self.logfile,'a') as f:
            f.write(str(args) + '\n')
    def log(self,msg):
        if self.print_log:
            print(msg)
        with open(self.logfile,'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + msg + '\n')

    def refresh(self,current,total,mode,msg=None):
        if current == 0:
            self.begin_time = time.time()
        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append("[{:>3d}/{:<3d}]".format(current+1,total))
        L.append(" {} |".format(mode.center(6)))
        L.append(" Step: {}".format(format_time(step_time).ljust(6)))
        L.append(" | Tot:{}".format(format_time(tot_time).ljust(8)))

        if msg:
            L.append(' | '+ msg)
        msg = ''.join(L)
        if current < total-1:
            print('\r',msg,end='')
        elif current == total-1:
            print('\r',msg)
            with open(self.logfile,'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\t'+msg+'\n')
        else:
            raise NotImplementedError
    
def  accuracy(output,target,topk=(1,)):
    """compute the accuracy over k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = torch.flatten(correct[:k],start_dim = 0).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100/batch_size))
        return res

def compute_correct(outputs: torch.tensor, targets: torch.tensor):
    _,predicted = outputs.max(1)
    return predicted.eq(targets).sum.item()

def get_output_label(model,x):
    outputs = model(x)
    _,predicted = outputs.max(1)
    return predicted

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad(requires_grad)






