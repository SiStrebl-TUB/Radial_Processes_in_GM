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
def compute_mmd(x, y):
    with torch.no_grad():
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        
        # 1. Median Heuristik berechnen (wie in der Referenz)
        # Für Effizienz sampeln wir oft nur eine Teilmenge oder nutzen x und y
        all_data = torch.cat([x, y], dim=0)
        
        # ACHTUNG: Bei 5000 Punkten (2500+2500) wird die cdist Matrix 5000x5000 groß.
        # Das braucht RAM, sollte aber auf modernen GPUs locker passen.
        dists = torch.cdist(all_data, all_data, p=2.0)
        
        # Median der quadratischen (!) Distanzen > 0 berechnen. 
        # (Die Referenz nutzt den Median der nicht-quadratischen Distanz und quadriert später, 
        # wir machen es direkt für den Nenner)
        sigma = torch.median(dists[dists > 0]).item()
        gamma = 1.0 / (2.0 * sigma**2)
        
        del all_data, dists
        
        # 2. Kernel mit dem dynamischen Gamma (1/2*sigma^2) berechnen
        def get_kernel(a, b):
            sq_dists = torch.cdist(a, b, p=2.0).pow(2)
            return torch.exp(-sq_dists * gamma)
            
        x_kernel = get_kernel(x, x)
        y_kernel = get_kernel(y, y)
        xy_kernel = get_kernel(x, y)
        
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        
        del x_kernel, y_kernel, xy_kernel
        gc.collect()
        
    return mmd.item() # Zurück als Float

@torch.no_grad()
def compute_trajectory_metrics(xs):
    """
    Berechnet die Transportkosten der generierten Trajektorien.
    Erwartet xs in der Form: (Zeit, Batch, Dimensionen)
    """
    # Differenz zwischen aufeinanderfolgenden Zeitschritten (dx)
    dx = xs[1:] - xs[:-1]
    
    # delta_t für das Integral
    num_steps = xs.shape[0] - 1
    dt = 1.0 / num_steps
    
    # 1. Path Length (Wie viele "Meter" ist der Vektor gewandert?)
    step_lengths = torch.norm(dx, p=2, dim=-1) # L2-Norm pro Schritt
    path_length = step_lengths.sum(dim=0).mean().item()
    
    # 2. Kinetic Energy (W2-Transportkosten = Integral ||v||^2 dt)
    sq_step_lengths = torch.sum(dx**2, dim=-1)
    kinetic_energy = (sq_step_lengths / dt).sum(dim=0).mean().item()
    
    # 3. Theoretisches Minimum (Gerade Linie von Start zu Ziel)
    straight_line_dist = torch.norm(xs[-1] - xs[0], p=2, dim=-1).mean().item()
    
    return path_length, kinetic_energy, straight_line_dist