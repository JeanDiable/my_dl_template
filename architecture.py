from torchvision.models.resnet import resnet18,resnet34,resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights,ResNet50_Weights
from collections import OrderedDict
import torch.nn.init as init
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from utils import *
import archs.cifar_resnet as cifar_resnet
config = load_config()
from archs import *

#to get a default architecture
def get_architecture(arch:str, dataset:str, pretrained:bool=False,normalize=False)->torch.nn.Module:
    if arch == "resnet18" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet18(pretrained=ResNet18_Weights.DEFAULT))
        cudnn.benchmark = True
    elif arch == "resnet34" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet34(pretrained=ResNet34_Weights.DEFAULT))
        cudnn.benchmark = True
    elif arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=ResNet50_Weights.DEFAULT))
        cudnn.benchmark = True
    
    if(pretrained):
        ckpt_path = config['ckpt']['cifar10']
    else:
        ckpt_path = None

    if arch == "resnet18" and dataset == "cifar10":
        model = cifar_resnet.resnet18(num_classes=10, ckpt_path=ckpt_path)
    elif arch == "resnet34" and dataset == "cifar10":
        model = cifar_resnet.resnet34(num_classes=10, ckpt_path=ckpt_path)
    elif arch == "resnet50" and dataset == "cifar10":
        model = cifar_resnet.resnet50(num_classes=10, ckpt_path=ckpt_path)
    elif "vgg" in arch:
        model = vgg(arch,ckpt_path = ckpt_path)
    else:
        raise Exception("Unknown architecture.")
    
    if normalize:
        return normalize_model(model,dataset)
    else:
        return model

#to normalize model
def normalize_model(model,dataset:str) -> torch.nn.Module:
    if "imagenet" in dataset:
        dataset = "imagenet"
    return NormalizedModel(model,normalize_map[dataset])

#the class which is used for normalizing model
class NormalizeModel(nn.Module):
    def __init__(self,net,data_normalize):
        super(NormalizeModel,self).__init__()
        self.net = net
        if data_normalize is None:
            self.data_normalize = None
        else:
            self.data_normalize = data_normalize
    
    def forward(self,x):
        if self.data_normalize is not None:
            x = self.data_normalize(x)
        return self.net(x)
    
class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self,mean,std):
        super(NormalizeByChannelMeanStd,self).__init__()
        if not isinstance(mean,torch.Tensor):
            mean = torch.ToTensor(mean)
        if not isinstance(std,torch.Tensor):
            std = torch.ToTensor(std)
        self.register_buffer("mean",mean)
        self.register_buffer("std",std)

    def forward(self,tensor):
        return self.normalize_fn(tensor,self.mean,self.std)
    
    def extra_repre(self):
        return "mean={},std={}".format(self.mean,self.std)
    
    def normalize_fn(self,tensor,mean,std):
        #we assume the color chanel is at dim = 1
        mean = mean[None,:,None,None]
        std = std[None,:,None,None]
        return tensor.sub(mean).div(std)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2470, 0.2435, 0.2616]

_MNIST_MEAN = [0.12486005]
_MNIST_STDDEV = [0.4898408]

normalize_map = {
    "cifar10" : NormalizeByChannelMeanStd(_CIFAR10_MEAN, _CIFAR10_STDDEV),
    "imagenet" : NormalizeByChannelMeanStd(_IMAGENET_MEAN, _IMAGENET_STDDEV),
    "mnist" : NormalizeByChannelMeanStd(_MNIST_MEAN, _MNIST_MEAN)
}

def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:] #remove'module.'
        new_state_dict[name] = v
    return new_state_dict

def init_params(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            init.kaiming_normal(m.weight,mode='fan_out')
            if m.bias:
                init.constant(m.bias,0)
        elif isinstance(m,nn.BatchNorm2d):
            init.constant(m.weight,1)
            init.constant(m.bias,0)
        elif isinstance(m,nn.Linear):
            init.normal(m.weight,std=1e-3)
            if m.bias:
                init.constant(m.bias,0)

if __name__ == "__main__":
    from dataset import get_dataloader
    from utils import accuracy
    [train_loader,_] = get_dataloader("cifar10",128)
    resnet18 = get_architecture("resnet18","cifar10",pretrained=True)

    for data in train_loader:
        x,y = data
        out = resnet18(x)
        print(accuracy(out,y))
