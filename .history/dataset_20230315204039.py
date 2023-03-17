from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import os
import torch
from typing import List
from torchvision.transforms import ToTensor
from utils import *

config = load_config()

#list of datasets
DATASETS = ["imagenet" , "cifar10", "cifar100", "COCO", "KITTI"]

train_map = {
    "cifar10":
        {
            "dataset_class":datasets.CIFAR10,
            "op": [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,4),
                transforms.ToTensor(),
            ],
            "mean" : [0.4914,0.4822,0.4465],
            "std" : [0.2470,0.2435,0.2616]
        },

    "cifar100":
        {
            "path":"/home/data/cifar100",
            "dataset_class":datasets.CIFAR100,
            "op": [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,4),
                transforms.ToTensor(),
            ],
            "mean": [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            "std":[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        },
    "imagenet":
        {
            "path":"/home/data/ILSVRC2012/train",
            "dataset_class":datasets.ImageFolder,
            "op": [
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],
            "mean" : [0.485, 0.456, 0.406],
            "std" : [0.229, 0.224, 0.225]
        }
}

test_map = {
    "cifar10":
        {
            "op": [
                transforms.ToTensor()
            ],
            "mean" : [0.4914,0.4822,0.4465],
            "std" : [0.2470,0.2435,0.2616],
        },
    
    "cifar100":
        {
            "path":"/home/data/cifar100",
            "op": [
                transforms.ToTensor(),
            ],
            "mean" : [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            "std":[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        },

    "imagenet":
        {
            "path":"/home/datasets/ILSVRC2012/train",
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
    train_trans = train_map[dataset]["op"]
    test_trans = test_map[dataset]["op"]

    if normalize:
        train_trans.append(transforms.Normalize(train_map[dataset]["mean"],train_map[dataset]["std"]))
        test_trans.append(transforms.Normalize(test_map[dataset]["mean"],test_map[dataset]["std"]))
    train_set = train_map[dataset]["dataset_class"](dataset_path,train = True,download = True,transform = transforms.Compose(train_trans))
    test_set = train_map[dataset]["dataset_class"](dataset_path,train = False,download = True,transform = transforms.Compose(test_trans))

    if shuffle:
        shuffles = [True, False]
    else:
        shuffles = [False, False]
    
    if isinstance(batch_size,int):
        batch_sizes = [batch_size for _ in range(2)]
    else:
        batch_sizes = batch_size

    sets = [train_set, test_set]
    return [DataLoader(sets[i],batch_size=batch_sizes[i],shuffle=shuffles[i],num_workers = num_workers) for i in range(2)]

if __name__ == "__main__":
    train_loader,test_loader = get_dataloader("cifar10",10)
