# code copied from 
# https://github.com/zacheberhart/Maximum-Mean-Discrepancy-Variational-Autoencoder/tree/master


import os
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import gc

@torch.no_grad()
def compute_kernel(x, y):
    with torch.no_grad():
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        result = torch.exp(-kernel_input) # (x_size, y_size)
        del kernel_input, tiled_x, tiled_y
        gc.collect()
    return result # (x_size, y_size)

@torch.no_grad()
def compute_mmd(x, y):
    with torch.no_grad():
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        del x_kernel, y_kernel, xy_kernel
        gc.collect()
    return mmd