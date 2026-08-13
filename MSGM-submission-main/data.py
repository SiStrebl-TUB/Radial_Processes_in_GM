

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

class PIV:
    def __init__(self, dim=1024, normalized=False, localized=False, few_data=False, ntrain_max=np.inf):
        self.dim = dim
        self.name = 'PIV'
        self.name += str(self.dim)
        
        # Namensgebung beibehalten, damit die MSGM Speicher-Ordner richtig benannt werden
        if localized:
            self.name += 'loc'
        if few_data:
            self.name += str(ntrain_max) + 'pts'
        if normalized:
            self.name += '_norm'

        # NEU: Wir zeigen auf DEINEN preprocessed Ordner
        folder = Path("data/preprocessed_piv")
        
        if not folder.exists():
            raise FileNotFoundError(f"Präprozessierte PIV Daten nicht gefunden in: {folder}. Lass zuerst preprocess_piv.py laufen!")

        print("Loading fast preprocessed PIV data from:", folder)
        
        # Blitzschnelles Laden der NPY-Dateien
        train_data = np.load(folder / "piv_train.npy")
        test_data = np.load(folder / "piv_test.npy")
        self.std = np.load(folder / "piv_std.npy")

        # Zurechtschneiden, falls dim nicht exakt 1024 ist (z.B. für kleinere Tests)
        self.npdata = train_data[:, :self.dim]
        self.npdatatest = test_data[:, :self.dim]
        self.std = self.std[:self.dim]

        # Normalisierung (Z-Score)
        if normalized:
            # Wichtig: Kleine Epsilon-Addition, falls std exakt 0 ist, um Division durch Null zu vermeiden
            self.npdata = self.npdata / (self.std + 1e-8)
            self.npdatatest = self.npdatatest / (self.std + 1e-8)

        # Attribute beibehalten, die das MSGM Skript eventuell abfragt
        self.max_nsamples = self.npdata.shape[0]
        self.max_nsamplestest = self.npdatatest.shape[0]

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
