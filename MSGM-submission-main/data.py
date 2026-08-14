

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from netCDF4 import Dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import random

from scipy.ndimage import gaussian_filter

class PIV:
    def __init__(self, dim=1024, normalized=False, localized=False, largeImage=True, smoothing=2, few_data=False, ntrain_max=np.inf):
        self.dim = dim
        self.name = 'PIV' + str(self.dim)
        
        if largeImage:
            self.name += 'largeIm'
            if smoothing == 1:
                self.name += 'smooth'
            if smoothing == 2:
                self.name += 'superSmooth'
            localized = True
            npixelx = int(np.sqrt(dim))  # 32
        elif localized:
            self.name += 'loc'
            
        if few_data:
            self.name += str(ntrain_max) + 'pts'
        if normalized:
            self.name += '_norm'

        # Zeige auf den "largerImage" Ordner mit den Original-Dateien der Autoren
        folder = Path(__file__).parent.parent / "data" / "largerImage" # Pfad evtl. anpassen!
        
        if not folder.exists():
            raise FileNotFoundError(f"Original-Datenordner nicht gefunden: {folder}")

        print(f"Loading authors' raw PIV data from: {folder}")

        # 1. Alle Einzeldateien laden und stapeln -> Form (Samples, 8192)
        files = sorted(folder.glob("*_vortdiv.npy"))
        if not files:
            raise FileNotFoundError(f"Keine _vortdiv.npy Dateien in {folder} gefunden!")
            
        npdata = np.vstack([np.load(f) for f in files])

        # 2. Skalieren und Zentrieren (wie Autoren)
        npdata = npdata / 2.5
        npdata = npdata - npdata.mean(axis=0)

        # 3. Reshape in 64x64 Gitter & Vorticity extrahieren
        npixelx_max = 64
        if largeImage:
            # WICHTIG: order='F' ist zwingend nötig, da DaVis/Matlab so exportiert!
            npdata = npdata.reshape((npdata.shape[0], npixelx_max, npixelx_max, 2), order='F')
            npdata = npdata[:, :, :, 0] # Nur Vorticity behalten -> Form (Samples, 64, 64)

            # 4. Smoothing (Smoothing=2 aus der Config)
            if smoothing > 0:
                print("Filtering images (Smoothing)...")
                if smoothing == 1:
                    sigmax = npdata.shape[1] // (3 * npixelx)
                elif smoothing == 2:
                    sigmax = npdata.shape[1] // npixelx
                    npdata *= 4  # Autoren-Multiplikator
                
                for i in range(npdata.shape[0]):
                    npdata[i, :, :] = gaussian_filter(npdata[i, :, :], sigma=sigmax)

            # 5. Subsampling auf Ziel-Dimension (z.B. 32x32 = 1024)
            print("Subsampling images to match required dimension...")
            ix = np.linspace(0, npdata.shape[1]-1, npixelx, dtype=int)
            iy = np.linspace(0, npdata.shape[2]-1, npixelx, dtype=int)
            npdata = npdata[:, ix, :]
            npdata = npdata[:, :, iy]

            # Wieder flachklopfen -> Form (Samples, 1024)
            npdata = npdata.reshape((npdata.shape[0], dim), order='F')
        else:
            npdata = npdata[:, 0:self.dim]

        # 6. Train/Test Split (exakt wie Autoren: n_test = 1/3)
        if few_data:
            n_train = min([2 * npdata.shape[0] // 3, ntrain_max])
            n_test = npdata.shape[0] - n_train 
        else:
            n_test = npdata.shape[0] // 3

        # Slice-Notation der Autoren
        self.npdata = npdata[0:-n_test, :]
        self.npdatatest = npdata[-n_test:, :]

        self.max_nsamples = self.npdata.shape[0]
        self.max_nsamplestest = self.npdatatest.shape[0]

        # 7. Standardabweichung berechnen und ggf. normalisieren
        self.std = npdata.std(axis=0)
        if normalized:
            # + 1e-8 um Division durch 0 zu vermeiden (sicherer als Original)
            self.npdata = self.npdata / (self.std + 1e-8)
            self.npdatatest = self.npdatatest / (self.std + 1e-8)

    def sample(self, n):               
        idx = np.random.randint(0, self.npdata.shape[0], size=n) 
        return torch.from_numpy(self.npdata[idx, :]).to(torch.float32)

    def sampletest(self, n):               
        idx = np.random.randint(0, self.npdatatest.shape[0], size=n) 
        return torch.from_numpy(self.npdatatest[idx, :]).to(torch.float32)
    
    def get_std(self):
        return torch.from_numpy(self.std).to(torch.float32)


class SwissRoll:
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """
    def __init__(self): 
        self.dim = 2
        self.name='swiss'
    def sample(self, n, noise=0.5):
        if noise is None:
            noise = 0.5
        return torch.from_numpy(
            make_swiss_roll(n, noise=noise)[0][:, [0, 2]].astype('float32') / 5.) # Changed: Pass noise as a keyword argument
    
    def sampletest(self, n, noise=0.5):
        return self.sample(n, noise)

class Cauchy:
    """
    multi-dimensional Cauchy distribution sampler.
    """
    def __init__(self, dim = 2, correlation = False, normalized = False):
        self.dim = dim
        self.name='cauchy' + str(self.dim)
        if correlation:
            self.A = torch.randn(dim, dim)
            self.name = self.name + "cor"
        else:
            self.A = torch.eye(dim)
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        if normalized:
            self.name = self.name + '_norm'
            self.A = torch.diag(1/self.std) @ self.A 
            cov = self.A @ self.A.T

        scale = (1.0/50)
        self.cauchy = torch.distributions.Cauchy(0.0, scale)

    def sample(self, n):
        return  self.cauchy.sample((n, self.dim)) @ self.A.T
    
    def sampletest(self, n):
        return self.sample(n)
    
    def get_std(self):
        return self.std
    

class Gaussian:
    """
    multi-dimensional Gaussian distribution sampler.
    """
    def __init__(self, dim = 2, correlation = True, normalized = False):
        self.dim = dim
        self.name='gaussian' + str(self.dim)
        if correlation:
            self.A = torch.randn(dim, dim)
            self.name = self.name + "cor"
        else:
            self.A = torch.eye(dim)
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        if normalized:
            self.name = self.name + '_norm'
            self.A = torch.diag(1/self.std) @ self.A 
            cov = self.A @ self.A.T
        self.normal = torch.distributions.Normal(0.0, 1.0)

    def sample(self, n):
        return  self.normal.sample((n, self.dim)) @ self.A.T
    
    def sampletest(self, n):
        return self.sample(n)
    
    def get_std(self):
        return self.std
