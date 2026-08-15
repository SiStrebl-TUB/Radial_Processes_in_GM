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
        # ZWINGEND ERFORDERLICH für 2500 Samples:
        # Konvertiere in 64-Bit, damit die winzigen MMD-Differenzen
        # beim anschließenden .mean() nicht abgerundet werden!
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        
        dim = float(x.size(1))
        
        # torch.cdist berechnet die Summe der quadratischen Distanzen.
        # Um die exakte Mathematik der Autoren (.mean(2) / dim) zu matchen,
        # müssen wir durch (dim * dim) teilen!
        kernel_input = torch.cdist(x, y, p=2.0).pow(2) / (dim * dim)
        
        result = torch.exp(-kernel_input)
        
        gc.collect()
    return result

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