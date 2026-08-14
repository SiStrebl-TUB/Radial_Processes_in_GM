# code copied and memory-optimized from 
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
def compute_kernel(x, y, chunk_size=1024):
    """
    Berechnet die Gauss-Kernel-Matrix speichereffizient in Chunks.
    Mathematisch 100% identisch zu:
    (tiled_x - tiled_y).pow(2).mean(2) / float(dim) -> exp(-kernel_input)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = float(x.size(1))
    
    # Skalierungsfaktor: .mean(2) / dim entspricht / (dim * dim)
    scale = 1.0 / (dim * dim)
    
    # Ergebnis-Tensor direkt auf demselben Device anlegen
    result = torch.empty((x_size, y_size), device=x.device, dtype=x.dtype)
    
    # Blockweise Berechnung, um OOM auf der GPU komplett zu verhindern
    for i in range(0, x_size, chunk_size):
        x_chunk = x[i : i + chunk_size]
        # cdist berechnet ||x_i - y_j||_2 direkt ohne 3D-Broadcasting
        dist_sq = torch.cdist(x_chunk, y, p=2.0).pow(2)
        result[i : i + chunk_size] = torch.exp(-dist_sq * scale)
        
    return result


@torch.no_grad()
def compute_mmd(x, y, chunk_size=1024):
    """
    Berechnet die MMD-Distanz ohne unnötigen VRAM-Overhead.
    """
    x_kernel = compute_kernel(x, x, chunk_size=chunk_size)
    y_kernel = compute_kernel(y, y, chunk_size=chunk_size)
    xy_kernel = compute_kernel(x, y, chunk_size=chunk_size)
    
    mmd = x_kernel.mean() + y_kernel.mean() - 2.0 * xy_kernel.mean()
    
    del x_kernel, y_kernel, xy_kernel
    if x.is_cuda:
        torch.cuda.empty_cache()
        
    return mmd