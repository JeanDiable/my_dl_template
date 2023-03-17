import os
import shutil
import torch
import yaml
import numpy as np
import time,datetime
from torchvision import transforms

def load_config() ->dict :
    with open("./config.yml") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config

