from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import os
import torch
from typing import List
from torchvision.transforms import ToTensor
from utils import *

cofig = load_config()

#list of datasets
DATASETS = ["imagenet" , "cifar10", "cifar100", "COCO", "KITTI"]

train_map = {
    "cifar10":
        {
            "dataset_class":datasets.CIFAR10,
            "op": [
                transforms.RandomHorizontalFlip,
                transforms.RandomCrop(32,4),
                transforms.ToTensor()
            ],
            "mean" : [0.4914,0.4822,0.4465],
            "std" : [0.2470,0.2435,0.2616]
        },
    "cifar100":
        {
            "path":"/home/data/cifar100"
            "dataset_class":datasets.CIFAR100,
            "image_op": [
                transforms.RandomHorizontalFlip,
                transforms.RandomCrop(32,4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ],
        },
    "imagenet":
        {
            "path":"/home/datasets/ILSVRC2012/train"
            "dataset_class":datasets.ImageFolder,
            "op": [
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],
            "mean" : [0.485, 0.456, 0.406],
            "std" : [0.229, 0.224, 0.225]
        },
}

test_map = {
    "cifar10":
        {
            "op": [
                transforms.ToTensor()
            ],
            "mean" : [0.4914,0.4822,0.4465],
            "std" : [0.2470,0.2435,0.2616]
        },
    "cifar100":
        {
            "path":"/home/data/cifar100"
            "image_op": [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ],
        },
    "imagenet":
        {
            "path":"/home/datasets/ILSVRC2012/train"
            "op": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ],
            "mean" : [0.485, 0.456, 0.406],
            "std" : [0.229, 0.224, 0.225]
        },
}

def get_dataloader(dataset : str, batch_size : int or list, shuffle : bool = False, normalize : bool = True, num_workers : int = 4) -> List[DataLoader]:
    """Return the dataloader as a Pytorch Dataset Object"""

    if dataset not in DATASETS:
        raise Exception("Unknown dataset")
    
    dataset_path = config["data"][dataset]