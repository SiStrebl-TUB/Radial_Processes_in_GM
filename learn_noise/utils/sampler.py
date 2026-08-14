from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math
import numpy as np
import torch
from torch.special import i0e, i1e

from pathlib import Path
from typing import Optional
from torch import Tensor
import pandas as pd

from sklearn.datasets import make_swiss_roll
from scipy.ndimage import gaussian_filter

try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - torchvision is optional
    datasets = None
    transforms = None
    DataLoader = None

Tensor = torch.Tensor


def _as_device_dtype(
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> Tuple[torch.device, torch.dtype]:
    if device is None:
        resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)
    if dtype is None:
        resolved_dtype = torch.get_default_dtype()
    else:
        resolved_dtype = dtype
    return resolved_device, resolved_dtype


def _log_normal_1d(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    var = std ** 2
    return -0.5 * ((x - mean) ** 2) / var - torch.log(std) - 0.5 * math.log(2 * math.pi)


def _logsumexp(a: Tensor, dim: int = -1) -> Tensor:
    return torch.logsumexp(a, dim=dim)


class BaseDistribution2D:
    """Interface for 2D distributions."""

    has_log_prob: bool = False

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Analytic log-density not available for this distribution.")


@dataclass
class CheckerboardStripes(BaseDistribution2D):
    low: float = -4.0
    high: float = 4.0

    has_log_prob: bool = True

    def _pick_square(self, n: int, device, dtype) -> Tensor:
        low_i = int(math.floor(self.low))
        high_i = int(math.floor(self.high))
        I = torch.arange(low_i, high_i, device=device)
        J = torch.arange(low_i, high_i, device=device)
        ii, jj = torch.meshgrid(I, J, indexing="ij")
        mask = ((ii + jj) % 2 == 0)
        valid = torch.stack([ii[mask], jj[mask]], dim=-1)
        idx = torch.randint(0, valid.shape[0], (n,), device=device)
        return valid[idx].to(dtype)

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        squares = self._pick_square(n, device, dtype)
        offs = torch.rand(n, 2, device=device, dtype=dtype)
        return squares + offs

    def log_prob(self, x: Tensor) -> Tensor:
        area_total = (self.high - self.low) ** 2
        log_const = -math.log(area_total / 2.0)
        i = torch.floor(x[..., 0])
        j = torch.floor(x[..., 1])
        inside = (x[..., 0] >= self.low) & (x[..., 0] <= self.high) & \
                 (x[..., 1] >= self.low) & (x[..., 1] <= self.high) & \
                 (((i + j) % 2) == 0)
        out = x.new_full(x.shape[:-1], float("-inf"))
        out[inside] = log_const
        return out


@dataclass
class GridGMM9(BaseDistribution2D):
    spacing: float = 1.0
    var: float = 0.0025
    weights: Optional[Sequence[float]] = None

    has_log_prob: bool = True

    def __post_init__(self):
        coords = (-float(self.spacing), 0.0, float(self.spacing))
        self._means = tuple((x, y) for x in coords for y in coords)
        if self.weights is None:
            w = [0.01, 0.1, 0.3, 0.2, 0.02, 0.15, 0.02, 0.15, 0.05]
        else:
            if len(self.weights) != len(self._means):
                raise ValueError(f"weights must have length {len(self._means)}")
            w = list(self.weights)
        total = sum(w)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = tuple(ww / total for ww in w)
        self._logw = None

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        weights = torch.tensor(self.weights, device=device, dtype=dtype)
        cat = torch.distributions.Categorical(probs=weights)
        idx = cat.sample((n,))
        means = torch.tensor(self._means, device=device, dtype=dtype)
        std = math.sqrt(self.var)
        noise = std * torch.randn(n, 2, device=device, dtype=dtype)
        return means[idx] + noise

    def log_prob(self, x: Tensor) -> Tensor:
        if self._logw is None or self._logw.device != x.device or self._logw.dtype != x.dtype:
            self._logw = torch.log(torch.tensor(self.weights, device=x.device, dtype=x.dtype))
        means = x.new_tensor(self._means)
        diff = x[:, None, :] - means[None, :, :]
        quad = (diff ** 2).sum(dim=-1) / self.var
        log_comp = -0.5 * (quad + 2 * math.log(2 * math.pi * self.var))
        return _logsumexp(self._logw + log_comp, dim=-1)


@dataclass
class NealFunnel2D(BaseDistribution2D):
    sigma1: float = 3.0
    alpha: float = 1.0

    has_log_prob: bool = True

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        x1 = self.sigma1 * torch.randn(n, 1, device=device, dtype=dtype)
        std2 = torch.exp(0.5 * self.alpha * x1)
        x2 = std2 * torch.randn(n, 1, device=device, dtype=dtype)
        return torch.cat([x1, x2], dim=-1) #x1

    def log_prob(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        lp1 = _log_normal_1d(x1, x1.new_tensor(0.0), x1.new_tensor(self.sigma1))
        var2 = torch.exp(self.alpha * x1)
        lp2 = -0.5 * (x2 ** 2) / var2 - 0.5 * (math.log(2 * math.pi) + self.alpha * x1)
        return lp1 + lp2#lp1


class ZScoreWrapper(BaseDistribution2D):
    def __init__(self, base: BaseDistribution2D, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = base
        self.mean = mean
        self.std = std
        self.has_log_prob = getattr(base, "has_log_prob", False)

    def to(self, device):
        device = torch.device(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        # Check if the underlying distribution has a .to() method (like our new fixes)
        if hasattr(self.base, "to"):
            self.base.to(device)
        return self

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        raw = self.base.sample(n, device=device, dtype=dtype)
        mean = self.mean.to(raw.device, raw.dtype)
        std = self.std.to(raw.device, raw.dtype)
        return (raw - mean) / std

    def log_prob(self, x: Tensor) -> Tensor:
        if not hasattr(self.base, "log_prob"):
            raise AttributeError("Wrapped sampler does not implement log_prob")
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        raw = x * std + mean
        log_det = torch.log(std.abs()).sum()
        return self.base.log_prob(raw) - log_det

    def to_raw(self, x: Tensor) -> Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return x * std + mean

    def __getattr__(self, attr):
        return getattr(self.base, attr)


class MNISTSampler:
    """Random batches from MNIST with optional flattening."""

    def __init__(
        self,
        *,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform=None,
        flatten: bool = True,
        preload_batch_size: int = 1024,
    ) -> None:
        if datasets is None or transforms is None or DataLoader is None:
            raise ImportError(
                "torchvision is required for the MNIST sampler but is not available"
            )

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        loader = DataLoader(dataset, batch_size=preload_batch_size, shuffle=False)
        data_chunks = []
        label_chunks = []
        for images, lbls in loader:
            if flatten:
                images = images.view(images.shape[0], -1)
            data_chunks.append(images)
            label_chunks.append(lbls)

        if not data_chunks:
            raise RuntimeError("MNIST dataset is empty or failed to load.")

        self.data = torch.cat(data_chunks, dim=0).contiguous()
        self.labels = torch.cat(label_chunks, dim=0).contiguous()
        self.flatten = flatten
        self.image_shape = (1, 28, 28)
        self.dim = self.data.shape[1] if flatten else self.data.shape[1:]

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data.shape[0], (n,))
        batch = self.data[idx].to(device=device, dtype=dtype)
        return batch

    def sample_with_labels(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data.shape[0], (n,))
        images = self.data[idx].to(device=device, dtype=dtype)
        label_tensor = self.labels[idx].to(device=device)
        return images, label_tensor


class CIFAR10Sampler:
    """Random batches from CIFAR-10 with optional flattening."""

    num_classes: int = 10

    def __init__(
        self,
        *,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform=None,
        flatten: bool = False,
        preload_batch_size: int = 1024,
    ) -> None:
        if datasets is None or transforms is None or DataLoader is None:
            raise ImportError(
                "torchvision is required for the CIFAR-10 sampler but is not available"
            )

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        loader = DataLoader(dataset, batch_size=preload_batch_size, shuffle=False)

        data_storage = []
        label_storage = []
        for images, labels in loader:
            if flatten:
                images = images.view(images.shape[0], -1)
            data_storage.append(images)
            label_storage.append(labels)

        self.data = torch.cat(data_storage, dim=0).contiguous()
        self.labels = torch.cat(label_storage, dim=0)
        self.flatten = bool(flatten)
        self.image_shape = (3, 32, 32)
        self.dim = self.data.shape[1] if self.flatten else self.data.shape[1:]

        # Cache per-class indices for deterministic subsets.
        class_indices = []
        for cls in range(self.num_classes):
            mask = torch.nonzero(self.labels == cls, as_tuple=False).view(-1)
            class_indices.append(mask)
        self.class_indices = class_indices
        self.class_counts = tuple(int(idx.shape[0]) for idx in class_indices)
        self.num_samples = int(self.data.shape[0])

    def _gather(
        self,
        idx: torch.Tensor,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = _as_device_dtype(device, dtype)
        batch = self.data[idx]
        labels = self.labels[idx]
        if dtype is not None and batch.dtype != dtype:
            batch = batch.to(dtype)
        return batch.to(device), labels.to(device)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        idx = torch.randint(0, self.data.shape[0], (n,), dtype=torch.long)
        batch, _ = self._gather(idx, device=device, dtype=dtype)
        return batch

    def sample_with_labels(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.data.shape[0], (n,), dtype=torch.long)
        return self._gather(idx, device=device, dtype=dtype)

    def sample_class_subset(
        self,
        cls: int,
        count: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= cls < self.num_classes):
            raise ValueError(f"Class index out of range: {cls}")
        pool = self.class_indices[cls]
        if count > int(pool.shape[0]):
            raise ValueError(
                f"Requested {count} samples for class {cls}, but only {int(pool.shape[0])} available"
            )
        choice = torch.randperm(pool.shape[0])[:count]
        idx = pool[choice]
        return self._gather(idx, device=device, dtype=dtype)


class TorchKacConstantSampler:
    """
    Mixture sampler for the 1D Kac displacement. Combines the atomic mass at
    ±c t with a continuous component tabulated via inverse CDF.
    """

    def __init__(
        self,
        a: float,
        c: float,
        T: float,
        M: int,
        K: int = 1024,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if T <= 0.0:
            raise ValueError("TorchKacConstantSampler requires T > 0.")
        if M <= 0 or K <= 0:
            raise ValueError("Lookup grid sizes M and K must be positive.")

        self.a = float(a)
        self.c = float(c)
        self.beta = self.a / self.c
        self.T = float(T)
        self.M = int(M)
        self.K = int(K)
        self.device, self.dtype = _as_device_dtype(device, dtype)

        t_grid = np.linspace(0.0, self.T, self.M + 1, dtype=np.float64)
        U = np.linspace(0.0, 1.0, self.K + 1, dtype=np.float64)
        X_table = np.zeros((self.M + 1, self.K + 1), dtype=np.float64)
        F_table = np.zeros((self.M + 1, self.K + 1), dtype=np.float64)

        table_device = torch.device("cpu")
        for j, t in enumerate(t_grid):
            ct = self.c * t
            if ct < 1e-16:
                X_table[j] = 0.0
                F_table[j] = U
                continue

            norm = -np.expm1(-self.a * t)
            r = np.sqrt(np.maximum(ct * ct - (ct * U) ** 2, 0.0))
            z = self.beta * r
            exp_fac = np.exp(z - self.a * t)

            z_t = torch.from_numpy(z).to(table_device, dtype=torch.float64)
            exp_fac_t = torch.from_numpy(exp_fac).to(table_device, dtype=torch.float64)

            term1 = self.beta * exp_fac_t * i0e(z_t)
            small_mask = z_t <= 1e-6
            series = 0.5 + (z_t ** 2) / 16.0 + (z_t ** 4) / 384.0
            ratio_t = torch.where(
                small_mask,
                series,
                i1e(z_t) / z_t,
            )
            term2 = self.beta * ct * exp_fac_t * ratio_t
            Kz = 0.5 * (term1 + term2)
            f = 2.0 * (Kz.cpu().numpy() / norm) * ct

            dU = U[1:] - U[:-1]
            F = np.empty(self.K + 1, dtype=np.float64)
            F[0] = 0.0
            F[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dU)
            if F[-1] > 0:
                F /= F[-1]
            else:
                F = U

            X_table[j] = ct * U
            F_table[j] = F

        quantiles = np.linspace(0.0, 1.0, self.K + 1, dtype=np.float64)
        invC = np.empty_like(X_table)
        for j in range(self.M + 1):
            invC[j] = np.interp(quantiles, F_table[j], X_table[j])

        self.t_grid = torch.tensor(t_grid, device=self.device, dtype=self.dtype)
        self.invC_table = torch.tensor(invC, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def sample(self, t: Tensor, dim: int = 1) -> Tensor:
        orig_shape = t.shape
        t_flat = t.reshape(-1).to(self.device).to(self.dtype)
        total = t_flat.numel()
        expanded = t_flat.unsqueeze(1).expand(-1, dim).reshape(-1)

        mix_u = torch.rand(total * dim, device=self.device, dtype=self.dtype)
        cont_u = torch.rand_like(mix_u)
        p0 = torch.exp(-self.a * expanded)
        is_atomic = mix_u < p0

        dt = self.T / self.M
        j = torch.clamp((expanded / dt).floor().long(), 0, self.M - 1)
        alpha = (expanded - self.t_grid[j]) / dt

        ut = torch.clamp(cont_u, max=(self.K - 1) / self.K) * self.K
        k = ut.floor().long()
        frac = ut - k

        x0 = self.invC_table[j, k]
        x1 = self.invC_table[j, k + 1]
        y0 = self.invC_table[j + 1, k]
        y1 = self.invC_table[j + 1, k + 1]

        xj = x0 + frac * (x1 - x0)
        xj1 = y0 + frac * (y1 - y0)
        x_cont = xj + alpha * (xj1 - xj)

        magnitude = torch.where(is_atomic, self.c * expanded, x_cont)
        signs = torch.where(torch.rand_like(magnitude) < 0.5, 1.0, -1.0).to(self.dtype)
        return (signs * magnitude).view(*orig_shape, dim)


class TorchQuantileSampler:
    """Quantile sampler for the MMD baseline flow."""

    def __init__(
        self,
        b: float,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if b <= 0:
            raise ValueError("Parameter b must be positive for TorchQuantileSampler.")
        self.b = float(b)
        self.device, self.dtype = _as_device_dtype(device, dtype)

    @torch.no_grad()
    def sample(self, t: Tensor, dim: int = 1) -> Tuple[Tensor, Tensor]:
        t = t.to(self.device).to(self.dtype)
        orig_shape = t.shape
        U = torch.rand(*orig_shape, dim, device=self.device, dtype=self.dtype)
        scale = self.b * (1.0 - torch.exp(-t / self.b))
        scale = scale[..., None]
        return (2.0 * U - 1.0) * scale, U
    
class MultiGaussianSampler:
    """Multivariate Gaussian sampler for baseline test case"""
    has_log_prob:bool = True

    def __init__(
        self,
        *,
        nCenter:int = 8,
        std:float = 0.25,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.nCenter = nCenter
        self.std = std
        self.device, self.dtype = _as_device_dtype(device, dtype)
        
    @torch.no_grad()
    def sample(self, batchsize,*, device = None, dtype = None):
        batch_per_center = int(batchsize/self.nCenter)
        if (batch_per_center + 0.1) < batchsize/self.nCenter:
            raise ValueError(f"Batchsize must be divisible by number of centers: {self.nCenter}")
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        angles = (torch.arange(self.nCenter, device=device, dtype=dtype) 
                  * (2 * math.pi / self.nCenter))
        std = self.std
        radius = std / (0.3 * math.sin(math.pi / self.nCenter))

        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        means = torch.stack([x, y], dim=1)
        return torch.normal(means.repeat(batch_per_center, 1), std)
    
    def log_prob(self, x):
        """
        x:       (B, 2)
        centers: (n, 2)
        """
        angles = (torch.arange(self.nCenter, device=self.device, dtype=self.dtype) 
                  * (2 * math.pi / self.nCenter))
        std = self.std
        radius = std / (0.3 * math.sin(math.pi / self.nCenter))

        x_centers = radius * torch.cos(angles)
        y_centers = radius * torch.sin(angles)
        means = torch.stack([x_centers, y_centers], dim=1)
        sigma = self.std
        diff = x[:, None, :] - means[None, :, :]   # (B, n, 2)
        sq_dist = (diff ** 2).sum(dim=2)              # (B, n)

        log_probs = -0.5 * sq_dist / sigma**2
        log_probs -= math.log(2 * math.pi * sigma**2)

        return torch.logsumexp(log_probs, dim=1) - math.log(means.shape[0])
    
class Donut:
    has_log_prob:bool = True

    def __init__(self, *, radius_mean=3, radius_std=0.5, device = None, dtype = None):
        """
        Erstellt eine torus-förmige Verteilung (Ring).
        
        Args:
            radius_mean: Der durchschnittliche Radius des Rings (Abstand von 0).
            radius_std: Die Dicke des Rings (Standardabweichung der radialen Normalverteilung).
        """
        self.radius_mean = radius_mean
        self.radius_std = radius_std
        
        # Wir nutzen eine normale Gaussian für den Radius
        self.radial_dist = torch.distributions.Normal(radius_mean, radius_std)

        self.device, self.dtype = _as_device_dtype(device, dtype)

    def sample(self, num_samples, *, device = None, dtype = None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        # 1. Ziehe zufällige Winkel (gleichverteilt)
        theta = torch.rand(num_samples) * 2 * math.pi
        
        # 2. Ziehe zufällige Radien (normalverteilt um radius_mean)
        r = self.radial_dist.sample((num_samples,))
        
        # 3. Konvertiere Polar -> Kartesisch
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return torch.stack([x, y], dim=1)

    def log_prob(self, points):
        """
        Berechnet die log likelihood p(x,y).
        points: Tensor der Form (N, 2)
        """
        x, y = points[:, 0], points[:, 1]
        
        # 1. Berechne den Radius r für jeden Punkt
        r = torch.sqrt(x**2 + y**2)
        
        # 2. Berechne log_prob des Radius unter der Normalverteilung
        # Das sagt uns, wie wahrscheinlich es ist, so weit weg vom Ursprung zu sein.
        lp_r = self.radial_dist.log_prob(r)
        
        # 3. Korrekturterm für Polarkoordinaten (Jacobian Determinante)
        # p(x,y) = p(r) * p(theta) / r
        # log p(x,y) = log p(r) + log(1/2pi) - log(r)
        
        # log(1 / 2pi) = -log(2pi)
        lp_theta = -math.log(2 * math.pi)
        
        # Der Jacobian Term: -log(r)
        # Wir addieren ein kleines Epsilon, falls r=0 ist (numerische Stabilität)
        jacobian = -torch.log(r + 1e-8)
        
        return lp_r + lp_theta + jacobian
    
class SnailDistribution:
    has_log_prob:bool = True

    def __init__(self,*, r_start = 3.5, r_end = 1.5, sigma = 0.3, device = None, dtype = None):
        """
        Erstellt eine Schnecken-Verteilung (eine Windung).
        r_start: Radius bei Winkel 0
        r_end: Radius bei Winkel 2pi
        sigma: Dicke der Verteilung (Streuung um die Spirale)
        """
        self.r_start = r_start
        self.r_end = r_end
        self.sigma = sigma

        # Wir brauchen eine Normalverteilung für den Fehlerterm (r - r_soll)
        self.noise_dist = torch.distributions.Normal(0, sigma)

        self.device, self.dtype = _as_device_dtype(device, dtype)
    def get_mean_radius(self, theta):
        """Berechnet den Soll-Radius für einen gegebenen Winkel."""
        # Normalisiere theta auf [0, 1]
        t = theta / (2 * math.pi)
        # Lineare Interpolation zwischen start und end
        return self.r_start + t * (self.r_end - self.r_start)

    def sample(self, num_samples, *, device = None, dtype = None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        # 1. Winkel ziehen (Uniform 0 bis 2pi)
        theta = torch.rand(num_samples) * 2 * math.pi
        
        # 2. Soll-Radius für diesen Winkel berechnen
        mu_r = self.get_mean_radius(theta)
        
        # 3. Rauschen hinzufügen (Dicke der Schnecke)
        r = mu_r + self.noise_dist.sample((num_samples,))
        
        # 4. Polar -> Kartesisch
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return torch.stack([x, y], dim=1)

    def log_prob(self, points):
        x, y = points[:, 0], points[:, 1]
        
        # 1. Radius r und Winkel theta aus Punkt rekonstruieren
        r_actual = torch.sqrt(x**2 + y**2)
        
        # atan2 liefert Werte zwischen -pi und +pi. 
        # Wir wollen aber 0 bis 2pi.
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2*math.pi, theta)
        
        # 2. Was 'sollte' der Radius an diesem Winkel sein?
        r_expected = self.get_mean_radius(theta)
        
        # 3. Die Abweichung (Residual) berechnen
        diff = r_actual - r_expected
        
        # 4. Log-Prob berechnen
        # p(x,y) = p(diff) * p(theta) / r
        
        # A) Wahrscheinlichkeit des Radius-Fehlers (Normalverteilung um 0)
        lp_diff = self.noise_dist.log_prob(diff)
        
        # B) Wahrscheinlichkeit des Winkels (Uniform 1/(2pi))
        lp_theta = -math.log(2 * math.pi)
        
        # C) Jacobian Korrektur (Polar -> Kartesisch): -log(r)
        jacobian = -torch.log(r_actual + 1e-8)
        
        return lp_diff + lp_theta + jacobian
    
class SpiralGenerator:
    has_log_prob:bool = False

    def __init__(self, *, n_rounds=3, r_start=0.0, r_end=5.0, sigma=0.2, device = None, dtype = None):
        """
        Generiert eine Archimedeische Spirale als nn.Module.
        
        Args:
            n_rounds (int): Anzahl der Umdrehungen (z.B. 3 * 360 Grad).
            r_start (float): Radius am Start (innen).
            r_end (float): Radius am Ende (außen).
            sigma (float): Standardabweichung des Rauschens (Dicke der Linie).
        """
        
        self.n_rounds = n_rounds
        self.r_start = r_start
        self.r_end = r_end
        self.sigma = sigma

        self.device, self.dtype = _as_device_dtype(device, dtype)

    def sample(self, n_samples, *, device = None, dtype = None):

        """
        Args:
            u (Tensor): Input Tensor.
                - Shape [Batch, 1]: Werte in [0, 1]. Steuert Position auf der Spirale. 
                  Rauschen wird intern generiert.
                - Shape [Batch, 2]: 
                  Dim 0 in [0, 1] steuert Position.
                  Dim 1 (Normalverteilt) steuert das Rauschen.
        
        Returns:
            Tensor [Batch, 2]: Die generierten (x, y) Punkte.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        u = torch.rand(n_samples, 1)
        # 1. Input aufsplitten oder Rauschen erzeugen
        if u.shape[1] == 1:
            # Nur Fortschritt gegeben -> Rauschen intern würfeln
            u_prog = u
            noise = torch.randn_like(u_prog)
        elif u.shape[1] == 2:
            # Fortschritt und Rauschen gegeben (voll deterministisch/kontrollierbar)
            u_prog = u[:, 0:1]
            noise = u[:, 1:2]
        else:
            raise ValueError(f"Input shape must be [Batch, 1] or [Batch, 2], got {u.shape}")

        # WICHTIG: u_prog sollte im Bereich [0, 1] liegen (wie bei Quantilsfunktion).
        # Wir clippen zur Sicherheit, falls mal was leicht drüber ist (numerik).
        u_prog = torch.clamp(u_prog, 0.0, 1.0)

        # 2. Density Correction (Wurzel ziehen)
        # Sorgt dafür, dass Punkte außen nicht dünner gesät sind als innen.
        # t läuft von 0 (Start) bis 1 (Ende)
        t = torch.sqrt(u_prog)

        # 3. Winkel berechnen
        # Gesamtwinkel = Anzahl Runden * 2pi
        max_phi = self.n_rounds * 2 * math.pi
        phi = t * max_phi

        # 4. Radius berechnen (Archimedeische Spirale: r ~ winkel)
        # Da t linear mit phi wächst (nach der Wurzel-Korrektur), können wir t nutzen.
        mu_r = self.r_start + t * (self.r_end - self.r_start)
        
        # Rauschen auf den Radius addieren
        r = mu_r + noise * self.sigma

        # 5. Polar -> Kartesisch
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)

        return torch.cat([x, y], dim=1)
    
class SimpleGaussianSampler:
    """Standard Gaussian sampler centered at 0 with configurable std."""
    has_log_prob: bool = True

    def __init__(
        self,
        *,
        dim: int = 2,
        std: float = 25.0,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dim = dim
        self.std = std
        self.device, self.dtype = _as_device_dtype(device, dtype)

    @torch.no_grad()
    def sample(self, batchsize, *, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
            
        # 1. Generate standard normal noise (mean=0, std=1)
        z = torch.randn(batchsize, self.dim, device=device, dtype=dtype)
        
        # 2. Scale by the desired standard deviation
        return z * self.std

    def log_prob(self, x):
        """
        Calculates log p(x) for a Gaussian centered at 0.
        x: (B, dim)
        """
        # The formula for a d-dimensional Gaussian log-prob:
        # -0.5 * d * log(2*pi) - d * log(sigma) - 0.5 * ||x||^2 / sigma^2
        
        const_term = -0.5 * self.dim * math.log(2 * math.pi)
        log_det_term = -self.dim * math.log(self.std)
        
        # Calculate squared Euclidean norm ||x||^2 per sample
        sq_norm = (x ** 2).sum(dim=1)
        data_term = -0.5 * sq_norm / (self.std ** 2)
        
        return const_term + log_det_term + data_term

class AnisotropicGaussianSampler:
    """
    Gaussian sampler centered at 0 with different standard deviations per dimension.
    """
    has_log_prob: bool = True

    def __init__(
        self,
        *,
        stds: List[float] = [50.0, 0.1], # Default: Long in dim 0, narrow in dim 1
        angle_deg: float = 45.0,         # Rotation in degrees
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.dim = len(stds)
        self.stds = torch.tensor(stds, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        # --- Create Rotation Matrix ---
        # Convert degrees to radians
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        
        # Standard 2D rotation matrix: [[cos, -sin], [sin, cos]]
        # We define it for row-vector multiplication (x @ R) usually requires transpose logic,
        # but let's define the forward rotation R.
        self.rotation_matrix = torch.tensor([
            [c, -s],
            [s,  c]
        ], device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def sample(self, batchsize, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 1. Generate axis-aligned samples (scaled by stds)
        # Shape: (B, 2)
        z = torch.randn(batchsize, self.dim, device=device, dtype=dtype)
        z_aligned = z * self.stds
        
        # 2. Rotate the samples
        # For row vectors x (B,2), rotation is x @ R.T
        return z_aligned @ self.rotation_matrix.T

    def log_prob(self, x):
        """
        x: (B, 2) in the rotated space
        """
        # 1. Rotate BACK to the axis-aligned space
        # Inverse of rotation R is R.T.
        # If forward was x @ R.T, inverse is x @ (R.T)^-1 = x @ R
        x_aligned = x @ self.rotation_matrix
        
        # 2. Compute log_prob as if it were axis-aligned
        # (The determinant of a rotation matrix is 1, so log_det doesn't change volume)
        const_term = -0.5 * self.dim * math.log(2 * math.pi)
        log_det_term = -torch.sum(torch.log(self.stds))
        
        normalized_x = x_aligned / self.stds
        data_term = -0.5 * torch.sum(normalized_x ** 2, dim=1)
        
        return const_term + log_det_term + data_term
    
class TwoSpirals:
    """
    Two interleaved spirals (Archimedean).
    Hard to model because of the empty space between arms.
    """
    has_log_prob: bool = False  # Analytische Dichte ist auf 2D Manifold schwer definiert

    def __init__(
        self,
        *,
        noise_std: float = 0.05,
        turns: float = 3.0,
        scale: float = 0.16,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.noise_std = noise_std
        self.turns = turns
        self.scale = scale

    @torch.no_grad()
    def sample(self, batchsize: int, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 50/50 Split zwischen den zwei Armen
        n_a = batchsize // 2
        n_b = batchsize - n_a

        # --- Arm 1 ---
        # sqrt(rand) sorgt für gleichmäßige Dichte entlang der Linie
        t_a = self.turns * 2 * math.pi * torch.sqrt(torch.rand(n_a, device=device, dtype=dtype))
        r_a = t_a * self.scale
        
        x_a = r_a * torch.cos(t_a)
        y_a = r_a * torch.sin(t_a)
        data_a = torch.stack([x_a, y_a], dim=1)
        
        # --- Arm 2 (rotiert um Pi) ---
        t_b = self.turns * 2 * math.pi * torch.sqrt(torch.rand(n_b, device=device, dtype=dtype))
        r_b = t_b * self.scale
        
        x_b = -r_b * torch.cos(t_b) 
        y_b = -r_b * torch.sin(t_b)
        data_b = torch.stack([x_b, y_b], dim=1)

        # Zusammenfügen
        data = torch.cat([data_a, data_b], dim=0)

        # Rauschen addieren
        noise = torch.randn_like(data) * self.noise_std
        data = data + noise
        
        # Shuffle, damit Batch nicht sortiert ist
        indices = torch.randperm(batchsize, device=device)
        return data[indices]
    
import torch
import math
from typing import Optional

class ThinAngles:
    def __init__(self, angles_deg: list[float] = [12.0, 30.0, 112.0, 200.0, 315.0], scales_long: list[float]=[1.5, 0.75, 2.5, 1.0, 2.0], *, scale_short: float = 0.01, device=None, dtype=torch.float32):
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.dim = 2
        self.num_components = len(angles_deg)
        self.log_mix_weight = math.log(1.0 / self.num_components)

        self.stds = torch.tensor([[l, scale_short] for l in scales_long], device=self.device, dtype=self.dtype)
        angles_rad = torch.tensor(angles_deg, device=self.device, dtype=self.dtype) * (math.pi / 180.0)
        cos_a, sin_a = torch.cos(angles_rad), torch.sin(angles_rad)
        self.rot_matrices = torch.stack([torch.stack([cos_a, -sin_a], dim=-1), torch.stack([sin_a, cos_a], dim=-1)], dim=-2)

    def to(self, device):
        self.device = torch.device(device)
        self.stds = self.stds.to(self.device)
        self.rot_matrices = self.rot_matrices.to(self.device)
        return self

    def sample(self, batchsize: int, *, device=None, dtype=None):
        dev = torch.device(device) if device else self.device
        dtype = dtype if dtype else self.dtype
        indices = torch.randint(0, self.num_components, (batchsize,), device=dev)
        z = torch.randn(batchsize, 2, device=dev, dtype=dtype)
        z_scaled = z * self.stds.to(dev)[indices]
        R_batch = self.rot_matrices.to(dev)[indices]
        return torch.einsum('nij, nj -> ni', R_batch, z_scaled)

    def log_prob(self, x):
        """
        Calculates log p(x) for the mixture of rotated, variable-length Gaussians.
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        const = -0.5 * self.dim * math.log(2 * math.pi)
        
        # log_det ist jetzt nicht mehr für alle gleich, sondern ein Vektor!
        # Shape: [Num_Components]
        log_det = -torch.sum(torch.log(self.stds), dim=-1)
        
        # Rückwärts-Rotation (wie zuvor)
        # Shape: [Batchsize, Num_Components, 2]
        x_aligned = torch.einsum('nj, kji -> nkj', x, self.rot_matrices)
        
        # PyTorch Broadcasting Magie: 
        # x_aligned [Batch, Komp, 2] geteilt durch self.stds [Komp, 2] funktioniert out-of-the-box!
        norm_x = x_aligned / self.stds
        data_term = -0.5 * torch.sum(norm_x**2, dim=-1) # Shape: [Batchsize, Num_Components]
        
        # Addieren der Terme. log_det wird automatisch auf alle Batches angewandt.
        log_probs_components = const + log_det + data_term
        
        return self.log_mix_weight + torch.logsumexp(log_probs_components, dim=1)
class ThinCross:
    """
    Mixture of two orthogonal, very thin Gaussians (Crux).
    Challenges models to represent disjoint/thin manifolds and the singularity at the center.
    """
    has_log_prob: bool = True

    def __init__(
        self,
        *,
        scale_long: float = 3.0,  # Länge der Arme
        scale_short: float = 0.01, # Dicke der Arme (sehr dünn)
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.dim = 2
        
        # Wir definieren die Parameter für das Gaussian Mixture Model (GMM)
        # Komponente 1: Horizontal (Große Var X, Kleine Var Y)
        self.std_hor = torch.tensor([scale_long, scale_short], device=self.device, dtype=self.dtype)
        
        # Komponente 2: Vertikal (Kleine Var X, Große Var Y)
        self.std_ver = torch.tensor([scale_short, scale_long], device=self.device, dtype=self.dtype)
        
        # Mixing Probability (log(0.5))
        self.log_mix_weight = math.log(0.5)

    @torch.no_grad()
    def sample(self, batchsize: int, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 1. Entscheiden, welcher Punkt zu welchem Arm gehört (Binomial/Bernoulli)
        # 0 = Horizontal, 1 = Vertikal
        mask = torch.bernoulli(torch.full((batchsize,), 0.5, device=device, dtype=dtype)).bool()
        
        # 2. Basis-Samples (Standard Normal)
        z = torch.randn(batchsize, 2, device=device, dtype=dtype)
        
        # 3. Skalieren basierend auf der Entscheidung
        # Wir bauen einen Tensor aus stds, der für jeden Sample passt
        stds = torch.zeros_like(z)
        stds[~mask] = self.std_hor.to(device) # Horizontal
        stds[mask]  = self.std_ver.to(device) # Vertikal
        
        return z * stds

    def log_prob(self, x):
        """
        Calculates log p(x) for the mixture.
        p(x) = 0.5 * N(x|hor) + 0.5 * N(x|ver)
        """
        # Sicherstellen, dass x auf richtigem Device ist
        if x.device != self.device:
            x = x.to(self.device)
            
        # Log-Prob für Komponente 1 (Horizontal)
        # log N(x; 0, diag(std_hor^2))
        const = -0.5 * self.dim * math.log(2 * math.pi)
        
        # Horizontal
        log_det_hor = -torch.sum(torch.log(self.std_hor))
        norm_x_hor = x / self.std_hor
        data_term_hor = -0.5 * torch.sum(norm_x_hor**2, dim=1)
        log_prob_hor = const + log_det_hor + data_term_hor
        
        # Vertikal
        log_det_ver = -torch.sum(torch.log(self.std_ver))
        norm_x_ver = x / self.std_ver
        data_term_ver = -0.5 * torch.sum(norm_x_ver**2, dim=1)
        log_prob_ver = const + log_det_ver + data_term_ver
        
        # Mixture Log-Prob: log( 0.5*exp(lp1) + 0.5*exp(lp2) )
        # = log(0.5) + log( exp(lp1) + exp(lp2) )
        # = log(0.5) + logsumexp([lp1, lp2])
        
        # Stacken für logsumexp
        stacked = torch.stack([log_prob_hor, log_prob_ver], dim=1)
        return self.log_mix_weight + torch.logsumexp(stacked, dim=1)

class RadialPareto:
    def __init__(self, r_min: float = 2.0, alpha: float = 2.5, device=None, dtype=torch.float32):
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.r_min = torch.tensor(r_min, device=self.device, dtype=self.dtype)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)

    def to(self, device):
        self.device = torch.device(device)
        self.r_min = self.r_min.to(self.device)
        self.alpha = self.alpha.to(self.device)
        return self

    def sample(self, batchsize: int, *, device=None, dtype=None):
        dev = torch.device(device) if device else self.device
        dtype = dtype if dtype else self.dtype
        u = torch.rand(batchsize, device=dev, dtype=torch.float64).clamp(min=1e-15)
        # Power operation on GPU
        r = self.r_min.to(dev).double() * torch.pow(u, -1.0 / self.alpha.to(dev).double())
        r = torch.clamp(r, max=1e6).to(dtype)
        theta_u = torch.rand(batchsize, device=dev, dtype=dtype) * 2 * math.pi
        theta = theta_u + 0.2 * torch.sin(2 * theta_u)
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    def log_prob(self, x):
        """
        Calculates log p(x) for the 2D Radial Pareto distribution.
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # Radius r berechnen
        r = x.norm(dim=1)
        
        # Maske für Punkte, die illegalerweise im leeren Zentrum liegen
        valid_mask = r >= self.r_min
        
        # 1. log_prob des Radius (1D Pareto PDF)
        # p(r) = (alpha * r_min^alpha) / r^(alpha + 1)
        log_p_r = (
            torch.log(self.alpha) 
            + self.alpha * torch.log(self.r_min) 
            - (self.alpha + 1.0) * torch.log(r)
        )
        
        # 2. Jacobian Korrektur: p(x,y) = p(r) / (2 * pi * r)
        log_p_cart = log_p_r - math.log(2 * math.pi) - torch.log(r)
        
        # Ungültige Radien (im Loch) bekommen log_prob = -inf
        result = torch.full_like(r, float('-inf'))
        result[valid_mask] = log_p_cart[valid_mask]
        
        return result

class IsotropicHeavyTail:
    """
    Isotropic distribution with heavy tails (2D Student's t-distribution).
    Looks like a Gaussian in the center but has a very slow, polynomial decay,
    producing extreme outliers. Perfect for testing radial tail augmentation.
    """
    has_log_prob: bool = True

    def __init__(
        self,
        *,
        scale: float = 1.0,     # Steuert die Breite des "Gauß-ähnlichen" Zentrums
        df: float = 2.0,        # Freiheitsgrade. Je kleiner, desto krasser die Tails! (df=1 ist Cauchy)
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Falls _as_device_dtype bei dir importiert wird, ansonsten hier manuell:
        if device is None: device = torch.device("cpu")
        if dtype is None: dtype = torch.float32
        self.device, self.dtype = torch.device(device), dtype
        
        self.scale = torch.tensor(scale, device=self.device, dtype=self.dtype)
        self.df = torch.tensor(df, device=self.device, dtype=self.dtype)

        # Vorberechnung der Konstante für die log_prob (2D Student-t)
        # Formel: log( Gamma((df+2)/2) / (Gamma(df/2) * df * pi * scale^2) )
        self.log_norm_const = (
            torch.lgamma((self.df + 2.0) / 2.0)
            - torch.lgamma(self.df / 2.0)
            - torch.log(self.df)
            - math.log(math.pi)
            - 2.0 * torch.log(self.scale)
        )

    @torch.no_grad()
    def sample(self, batchsize: int, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # Ein eleganter Trick der Statistik: Man kann eine Student-t Verteilung ziehen,
        # indem man eine Standard-Gauß-Verteilung durch die Wurzel einer Chi-Quadrat-Verteilung teilt!
        
        # 1. Normale 2D Gauß-Richtung & Radius-Basis
        z = torch.randn(batchsize, 2, device=device, dtype=dtype)
        
        # 2. Der Heavy-Tail Generator (Chi-Quadrat)
        chi2_sampler = torch.distributions.Chi2(self.df)
        v = chi2_sampler.sample((batchsize, 1)).to(device=device, dtype=dtype)
        
        # 3. Mischen: Wenn 'v' winzig wird (was bei Chi2 oft passiert), 
        # explodiert der Faktor und schießt den Punkt weit, weit nach außen!
        scaling_factor = self.scale * torch.sqrt(self.df / v)
        
        x = z * scaling_factor
        return x

    def log_prob(self, x):
        """
        Calculates log p(x) for the 2D Student's t-distribution.
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # Quadrierter Radius r^2 = x^2 + y^2
        r2 = torch.sum(x**2, dim=1)
        
        # Variabler Teil der Log-Wahrscheinlichkeit:
        # Formel: -((df + 2) / 2) * log(1 + r^2 / (df * scale^2))
        # (Wir nutzen log1p für höhere numerische Stabilität bei kleinen Radien)
        var_part = -((self.df + 2.0) / 2.0) * torch.log1p(r2 / (self.df * self.scale**2))
        
        return self.log_norm_const + var_part
    
class TwoRings:
    """
    Distribution of two concentric rings with different radii.
    Good for testing mode collapse (does it drop the outer ring?) 
    and disconnected manifolds.
    """
    has_log_prob: bool = True

    def __init__(
        self,
        *,
        radius_1: float = 1.0,
        radius_2: float = 3.0,
        std: float = 0.1,  # Dicke der Ringe (Radial noise)
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        
        # Parameter als Tensoren speichern
        self.r1 = torch.tensor(radius_1, device=self.device, dtype=self.dtype)
        self.r2 = torch.tensor(radius_2, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)
        
        # Wir gehen von einer 50/50 Mischung aus (log(0.5))
        self.log_mix_weight = math.log(0.5)

    @torch.no_grad()
    def sample(self, batchsize: int, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 1. Auswahl: Welcher Ring? (Bernoulli 50%)
        # mask=0 -> Ring 1, mask=1 -> Ring 2
        mask = torch.bernoulli(torch.full((batchsize,), 0.5, device=device, dtype=dtype)).bool()
        
        # 2. Winkel sampeln (Gleichverteilt 0 bis 2pi)
        theta = torch.rand(batchsize, device=device, dtype=dtype) * 2 * math.pi
        
        # 3. Radius sampeln (Normalverteilt um den jeweiligen Ring-Radius)
        # Wir bereiten einen Tensor mit den "Target-Radien" vor
        target_r = torch.zeros(batchsize, device=device, dtype=dtype)
        target_r[~mask] = self.r1
        target_r[mask]  = self.r2
        
        # r ~ N(target_r, std)
        r = torch.randn(batchsize, device=device, dtype=dtype) * self.std + target_r
        
        # 4. Polar -> Kartesisch
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return torch.stack([x, y], dim=1)

    def log_prob(self, x):
        """
        Calculates log p(x).
        Uses radial gaussian approximation + Jacobian correction for polar coords.
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # Radius r jedes Punktes berechnen
        r = x.norm(dim=1) + 1e-8 # eps für numerische Stabilität bei log(r)
        
        # Konstanten für log_prob einer 1D-Normalverteilung
        # log_prob_norm = -0.5 * log(2pi) - log(std) - 0.5 * ((x-mu)/std)^2
        const_norm = -0.5 * math.log(2 * math.pi) - torch.log(self.std)
        
        # --- Log Prob Ring 1 ---
        # Radialer Anteil: Wie weit weicht r von r1 ab?
        log_p_rad1 = const_norm - 0.5 * ((r - self.r1) / self.std) ** 2
        # Jacobian Anteil: Transformation von Polar (r, theta) zu Kartesisch (x,y)
        # p(x,y) = p(r) * p(theta) / r
        # log p(x,y) = log p(r) + log(1/2pi) - log(r)
        log_p_cart1 = log_p_rad1 - math.log(2 * math.pi) - torch.log(r)
        
        # --- Log Prob Ring 2 ---
        log_p_rad2 = const_norm - 0.5 * ((r - self.r2) / self.std) ** 2
        log_p_cart2 = log_p_rad2 - math.log(2 * math.pi) - torch.log(r)
        
        # --- Mixture ---
        # log( 0.5 * exp(L1) + 0.5 * exp(L2) )
        stacked = torch.stack([log_p_cart1, log_p_cart2], dim=1)
        return self.log_mix_weight + torch.logsumexp(stacked, dim=1)
    
@dataclass
class RingGMM(BaseDistribution2D):
    radius_inner: float = 1.0
    radius_outer: float = 3.0
    std: float = 0.15
    device: Optional[torch.device | str] = None
    dtype: torch.dtype = torch.float32

    has_log_prob: bool = True

    def __post_init__(self):
        self.device, self.dtype = _as_device_dtype(self.device, self.dtype)
        self.std_tensor = torch.tensor(self.std, device=self.device, dtype=self.dtype)
        
        means = []
        log_weights = []
        # Inner Ring (3)
        w_inner = math.log(0.5 / 3.0)
        for i in range(3):
            theta = (2 * math.pi / 3) * i
            means.append([self.radius_inner * math.cos(theta), self.radius_inner * math.sin(theta)])
            log_weights.append(w_inner)
        # Outer Ring (5)
        w_outer = math.log(0.5 / 5.0)
        offset = math.pi / 5 
        for i in range(5):
            theta = (2 * math.pi / 5) * i + offset
            means.append([self.radius_outer * math.cos(theta), self.radius_outer * math.sin(theta)])
            log_weights.append(w_outer)

        self.means = torch.tensor(means, device=self.device, dtype=self.dtype)
        self.log_weights = torch.tensor(log_weights, device=self.device, dtype=self.dtype)

    def to(self, device):
        self.device = torch.device(device)
        self.means = self.means.to(self.device)
        self.log_weights = self.log_weights.to(self.device)
        self.std_tensor = self.std_tensor.to(self.device)
        return self

    def sample(self, batchsize: int, *, device=None, dtype=None) -> Tensor:
        # Use provided device or fall back to self.device
        dev = torch.device(device) if device else self.device
        dtype = dtype if dtype else self.dtype
        
        probs = torch.exp(self.log_weights.to(dev))
        mode_indices = torch.multinomial(probs, batchsize, replacement=True)
        
        sample_means = self.means.to(dev)[mode_indices]
        epsilon = torch.randn(batchsize, 2, device=dev, dtype=dtype)
        return sample_means + epsilon * self.std_tensor.to(dev)

    def log_prob(self, x):
        """
        Calculates log p(x) for the Mixture of Gaussians.
        log p(x) = log sum_k ( w_k * N(x | mu_k, sigma) )
                 = logsumexp_k ( log(w_k) + log N(x | ...) )
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # x shape: (B, 2)
        # means shape: (8, 2)
        # Wir brauchen shape (B, 8, 2) für Broadcasting, um Distanz zu JEDEM Mode zu berechnen
        x_expanded = x.unsqueeze(1)       # (B, 1, 2)
        means_expanded = self.means.unsqueeze(0) # (1, 8, 2)
        
        # Quadratische Differenz (Distanz zum Quadrat)
        # sum über die Koordinaten (dim 2) -> (B, 8)
        diff_sq = torch.sum((x_expanded - means_expanded)**2, dim=2)
        
        # Log-Prob für jeden Modus (ohne Gewicht)
        # log N = -0.5*log(2pi) - log(std) - 0.5 * (dist^2 / std^2)
        # Da wir 2D sind (dim=2): -log(2pi) - 2*log(std) ... (Faktor 0.5 fällt bei log det weg für d=2)
        # Exakte Formel 2D isotrop: -log(2*pi*sigma^2) - 0.5 * dist^2 / sigma^2
        const = -math.log(2 * math.pi) - 2 * torch.log(self.std)
        log_probs_per_mode = const - 0.5 * (diff_sq / (self.std ** 2))
        
        # Gewichte addieren: log(w_k) + log p(x|k)
        weighted_log_probs = self.log_weights.unsqueeze(0) + log_probs_per_mode
        
        # LogSumExp über alle Modi (dim 1)
        return torch.logsumexp(weighted_log_probs, dim=1)
    

class RareOutlierGMM:
    """
    A mixture of 3 tight Gaussians near the origin and 1 extremely rare outlier far away.
    Challenges the model to capture low-probability modes without creating artifacts (bridges).
    """
    has_log_prob: bool = True
    def __init__(
            self,
            *,
            radius_inner: float = 0.5,
            radius_outer: float = 3.0,
            std: float = 0.05,        # Sehr klein, damit sie sich nicht überlappen
            outlier_prob: float = 0.001, # 1 aus 1000 Samples
            device: Optional[torch.device | str] = None,
            dtype: torch.dtype = torch.float32,
        ) -> None:
            self.device, self.dtype = _as_device_dtype(device, dtype)
            self.std = torch.tensor(std, device=self.device, dtype=self.dtype)
            
            # --- Definition der 4 Modi ---
            means = []
            weights = []

            # 1. Die 3 inneren Gaussians (Radius 0.5)
            # Teilen sich die verbleibende Wahrscheinlichkeit (0.999)
            prob_inner = (1.0 - outlier_prob) / 3.0
            
            for i in range(3):
                theta = (2 * math.pi / 3) * i  # 0, 120, 240 Grad
                x = radius_inner * math.cos(theta)
                y = radius_inner * math.sin(theta)
                means.append([x, y])
                weights.append(prob_inner)

            # 2. Der seltene Ausreißer (Radius 3, unten rechts)
            theta_out = -math.pi / 4 # -45 Grad (unten rechts)
            x_out = radius_outer * math.cos(theta_out)
            y_out = radius_outer * math.sin(theta_out)
            
            means.append([x_out, y_out])
            weights.append(outlier_prob)

            # Tensoren erstellen
            self.means = torch.tensor(means, device=self.device, dtype=self.dtype) # (4, 2)
            # Wir speichern log_weights für numerisch stabile log_prob Berechnung
            self.log_weights = torch.tensor(weights, device=self.device, dtype=self.dtype).log()
            self.num_modes = 4

    @torch.no_grad()
    def sample(self, batchsize: int, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 1. Modi auswählen
        # Wir nutzen multinomial auf den Wahrscheinlichkeiten (exp der log_weights)
        probs = torch.exp(self.log_weights)
        mode_indices = torch.multinomial(probs, batchsize, replacement=True) # (B,)
        
        # 2. Mittelpunkte holen
        selected_means = self.means[mode_indices] # (B, 2)
        
        # 3. Gaussian Noise addieren
        # Alle haben dieselbe Standardabweichung
        epsilon = torch.randn(batchsize, 2, device=device, dtype=dtype)
        samples = selected_means + epsilon * self.std
        
        return samples

    def log_prob(self, x):
        """
        GMM Log Prob:
        log p(x) = log sum_k ( w_k * N(x | mu_k, sigma) )
                = logsumexp( log(w_k) + log_prob_k(x) )
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # x: (B, 2) -> (B, 1, 2)
        # means: (4, 2) -> (1, 4, 2)
        x_expanded = x.unsqueeze(1)
        means_expanded = self.means.unsqueeze(0)
        
        # Distanz zum Quadrat berechnen (B, 4)
        diff_sq = torch.sum((x_expanded - means_expanded)**2, dim=2)
        
        # Log-Prob für jeden der 4 Modi berechnen
        # const = -log(2pi) - 2*log(std) (für 2D)
        const = -math.log(2 * math.pi) - 2 * torch.log(self.std)
        log_probs_per_mode = const - 0.5 * (diff_sq / (self.std ** 2))
        
        # Gewichtung addieren
        # log(w_k * p_k) = log(w_k) + log(p_k)
        weighted_log_probs = self.log_weights.unsqueeze(0) + log_probs_per_mode
        
        # Summenbildung im Log-Raum
        return torch.logsumexp(weighted_log_probs, dim=1)
    
class WeirdGMM:
    """
    5 Modes Total:
    - 3 Inner Gaussians (shifted slightly from origin)
    - 1 Rare Outlier (Bottom-Right, prob=0.01)
    - 1 Normal 'Counter-Balance' (Top-Left, opposite the outlier)
    """
    has_log_prob: bool = True
    
    def __init__(
        self,
        *,
        radius_inner: float = 0.5,
        radius_outer: float = 3.0,
        std: float = 0.05,
        outlier_prob: float = 0.01, # Changed to 1/100
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)
        
        means = []
        weights = []

        # --- Probability Math ---
        # Total = 1.0
        # Outlier = 0.01
        # Remaining = 0.99
        # We have 3 inner modes + 1 opposite mode = 4 "Normal" modes
        prob_normal = (1.0 - outlier_prob) / 4.0

        # 1. The 3 Inner Gaussians (The central shift)
        # Shifted by radius_inner from (0,0)
        for i in range(3):
            theta = (2 * math.pi / 3) * i 
            x = radius_inner * math.cos(theta)
            y = radius_inner * math.sin(theta)
            means.append([x, y])
            weights.append(prob_normal)

        # 2. The Rare Outlier (Bottom-Right)
        theta_out = -math.pi / 4  # -45 degrees
        x_out = radius_outer * math.cos(theta_out)
        y_out = radius_outer * math.sin(theta_out)
        means.append([x_out, y_out])
        weights.append(outlier_prob)

        # 3. The Opposite "Normal" Gaussian (Top-Left)
        # Directly opposite the outlier (theta + pi)
        theta_opp = theta_out + math.pi # 135 degrees
        x_opp = radius_outer * math.cos(theta_opp)
        y_opp = radius_outer * math.sin(theta_opp)
        means.append([x_opp, y_opp])
        weights.append(prob_normal)

        # Finalize Tensors
        self.means = torch.tensor(means, device=self.device, dtype=self.dtype) 
        self.log_weights = torch.tensor(weights, device=self.device, dtype=self.dtype).log()
        self.num_modes = 5

    @torch.no_grad()
    def sample(self, batchsize: int, *, device = None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        # 1. Select Modes
        probs = torch.exp(self.log_weights)
        mode_indices = torch.multinomial(probs, batchsize, replacement=True) 
        
        # 2. Get Means
        selected_means = self.means[mode_indices] 
        
        # 3. Add Noise
        epsilon = torch.randn(batchsize, 2, device=self.device, dtype=self.dtype)
        samples = selected_means + epsilon * self.std
        return samples

    def log_prob(self, x):
        if x.device != self.device: x = x.to(self.device)
            
        x_expanded = x.unsqueeze(1)       # (B, 1, 2)
        means_expanded = self.means.unsqueeze(0) # (1, 5, 2)
        
        diff_sq = torch.sum((x_expanded - means_expanded)**2, dim=2)
        const = -math.log(2 * math.pi) - 2 * torch.log(self.std)
        log_probs_per_mode = const - 0.5 * (diff_sq / (self.std ** 2))
        
        weighted_log_probs = self.log_weights.unsqueeze(0) + log_probs_per_mode
        return torch.logsumexp(weighted_log_probs, dim=1)
    
class QuantileOnlySampler:
    """
    Simple 1D sampler for testing quantile.
    No log_prob calculation included.
    """
    has_log_prob: bool = False

    def __init__(
        self,
        *,
        mean: float = 0.0,  # Globales Zentrum (bei 0 sind Peaks bei -1 und 1)
        std: float = 0.3,   # Breite der einzelnen Hügel
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = _as_device_dtype(device, dtype)
        self.std = std
        self.global_mean = mean

    @torch.no_grad()
    def sample(self, batchsize, *, device=None, dtype=None):
        if device is None: device = self.device
        if dtype is None: dtype = self.dtype
        
        # 1. Wähle linken oder rechten Hügel (50/50 Chance)
        # 0 -> linker Hügel, 1 -> rechter Hügel
        choices = torch.randint(0, 2, (batchsize, 1), device=device).to(dtype)
        
        # 2. Berechne die lokalen Means für jeden Punkt
        # choices=0 => offset -1
        # choices=1 => offset +1
        # Formel: 2*c - 1 macht aus {0,1} -> {-1, 1}
        offsets = 2 * choices - 1
        
        # 3. Das eigentliche Zentrum bestimmen
        # Peaks bei: global_mean - 1  UND  global_mean + 1
        local_means = self.global_mean + offsets

        # 4. Noise hinzufügen
        noise = torch.randn(batchsize, 1, device=device, dtype=dtype)
        return local_means + noise * self.std
    
class EmpiricalNormSampler:
    """
    Samples norms based on the Empirical CDF of the training data.
    """
    def __init__(
        self,
        training_vectors: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        """
        training_vectors: (M, D) tensor of training data
        """
        self.epsilon = epsilon
        self.device = training_vectors.device
        self.dtype = training_vectors.dtype

        # 1. Calculate Norms ||x|
        norms = torch.norm(training_vectors, dim=1)

        # 2. Calculate Log-Norms R (The values defining F_hat)
        # Based on Eq (C.3): ||x|| = exp(R) - eps  =>  R = log(||x|| + eps)
        self.R_values = torch.log(norms + epsilon)

        # 3. SORT them to create the inverse CDF lookup table
        # F^-1(u) is literally just looking up the value at index u*M
        self.sorted_R, sorted_idcs = torch.sort(self.R_values)
        outliers = int(0.0001 * self.sorted_R.shape[0])  # Keep track of top 5% outliers
        self.outliers = training_vectors[sorted_idcs[-outliers:]]  # Optional: Keep track of outliers (largest norms)
        self.M = self.sorted_R.shape[0]
        """zero_diff_mask = (self.sorted_R[1:] - self.sorted_R[:-1]) == 0.0
        duplicate_radii = self.sorted_R[1:][zero_diff_mask]

        # Zähle, wie viele UNTERSCHIEDLICHE Radien dieses Problem haben
        unique_duplicates, counts = torch.unique(duplicate_radii, return_counts=True)

        print(f"Es gibt {len(unique_duplicates)} spezifische Radien, die exakt mehrfach vorkommen.")

        # Zeige die Top 5 der am stärksten betroffenen Radien
        # (Sortiere nach Häufigkeit)
        sorted_counts, sort_idx = torch.sort(counts, descending=True)
        print("Die extremsten Fälle:")
        for i in range(min(5, len(sorted_counts))):
            idx = sort_idx[i]
            print(f"Radius {unique_duplicates[idx].item()}: {sorted_counts[i].item() + 1} Punkte liegen exakt hier!")  # Debug: Check spacing of R values"""

    @torch.no_grad()
    def sample_norms(self, batch_size: int) -> torch.Tensor:
        """
        Returns sampled norms ||x_T||
        """
        # A. Implicitly sample u ~ U(0,1) and find F^-1(u)
        # We do this by picking random indices [0, M-1]
        # This is much faster than generating floats and searching.
        indices = torch.randint(
            low=0, 
            high=self.M, 
            size=(batch_size,), 
            device=self.device
        )
        
        # B. Lookup the log-norms (R values)
        # This corresponds to F_log_inverse(u) in Eq (C.3)
        sampled_R = self.sorted_R[indices]

        # C. Transform back to norm space (Eq C.3)
        # ||x_T|| = exp(R) - epsilon
        sampled_norms = torch.exp(sampled_R) - self.epsilon
        
        return sampled_norms
    
    @torch.no_grad()
    def get_heavy_tail_samples(self, num_samples: int) -> torch.Tensor:
        """
        Optional: Get samples from the heavy tail (largest norms).
        This can be used for testing how well the model captures outliers.
        """
        if self.outliers.shape[0] == 0:
            raise ValueError("No outliers found in the training data.")
        
        # Randomly sample from the outliers
        indices = torch.randint(
            low=0, 
            high=self.outliers.shape[0], 
            size=(num_samples,), 
            device=self.device
        )
        return self.outliers[indices]
    
    
class InterpolatedNormSampler:
    """
    Samples norms based on the Empirical CDF of the training data,
    but uses linear interpolation to generate values BETWEEN data points.
    """
    def __init__(
        self,
        training_vectors: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        """
        training_vectors: (M, D) tensor of training data
        epsilon: The epsilon used in Eq (C.3)
        """
        self.epsilon = epsilon
        self.device = training_vectors.device
        self.dtype = training_vectors.dtype

        # 1. Berechne Normen ||x||
        norms = torch.linalg.norm(training_vectors, dim=1)

        # 2. Berechne Log-Normen R = log(||x|| + eps)
        self.R_values = torch.log(norms + epsilon)

        # 3. Sortieren ist notwendig für die Quantils-Interpolation
        self.sorted_R, _ = torch.sort(self.R_values)
        self.M = self.sorted_R.shape[0]

    @torch.no_grad()
    def sample_norms(self, batch_size: int) -> torch.Tensor:
        """
        Returns sampled norms using linear interpolation.
        """
        # 1. Ziehe zufällige Positionen im Array (Floats statt Ints)
        # Bereich: [0, M-1]
        # u ist der fließende Index, wo wir interpolieren wollen
        u_indices = torch.rand(batch_size, device=self.device) * (self.M - 1)
        
        # 2. Finde die benachbarten Indizes (Links und Rechts)
        idx_floor = u_indices.floor().long()  # Unterer Nachbar
        idx_ceil = idx_floor + 1              # Oberer Nachbar
        
        # Sicherheitshalber clippen (falls u_indices durch Rundung genau M-1 trifft)
        # idx_ceil darf maximal M-1 sein
        idx_ceil = torch.clamp(idx_ceil, max=self.M - 1)

        # 3. Berechne das Gewicht für die Interpolation
        # alpha ist der Anteil, wie nah wir am rechten Nachbarn sind (zwischen 0 und 1)
        alpha = u_indices - idx_floor.float()
        
        # 4. Werte aus der sortierten Liste holen
        val_left = self.sorted_R[idx_floor]
        val_right = self.sorted_R[idx_ceil]
        
        # 5. Lineare Interpolation im Log-Raum (R-Werte)
        # R_neu = (1-alpha)*Links + alpha*Rechts
        sampled_R = (1 - alpha) * val_left + alpha * val_right
        
        # 6. Zurückrechnen auf Normen
        # ||x_T|| = exp(R_neu) - epsilon
        sampled_norms = torch.exp(sampled_R) - self.epsilon
        
        return sampled_norms
    

# Stelle sicher, dass _as_device_dtype und BaseDistribution2D importiert/verfügbar sind.

class MSGM_SwissRoll(BaseDistribution2D):
    has_log_prob: bool = False

    def __init__(self, noise: float = 0.5):
        self.noise = noise

    def to(self, device):
        # SwissRoll speichert keine Tensoren im State, wir geben einfach self zurück
        return self

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        # Exakt die Logik aus MSGM:
        data = make_swiss_roll(n, noise=self.noise)[0][:, [0, 2]].astype('float32') / 5.0
        return torch.tensor(data, device=device, dtype=dtype)


class MSGM_Gaussian(BaseDistribution2D):
    has_log_prob: bool = True

    def __init__(self, dim: int = 16, correlation: bool = True, normalized: bool = False, seed: int = 0):
        self.dim = dim
        
        # WICHTIG: Um exakt denselben Datensatz wie MSGM zu generieren, MÜSSEN wir
        # den RNG-Seed für die Erstellung der Matrix A fixieren (im MSGM-Skript ist es 0).
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        
        if correlation:
            self.A = torch.randn(dim, dim)
        else:
            self.A = torch.eye(dim)
            
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        
        if normalized:
            self.A = torch.diag(1.0 / self.std) @ self.A 
            cov = self.A @ self.A.T
            
        # Seed wiederherstellen, damit das restliche Training nicht beeinflusst wird
        torch.set_rng_state(rng_state)

        # Für log_prob
        self.cov = cov
        self.mean = torch.zeros(dim)

    def to(self, device):
        device = torch.device(device)
        self.A = self.A.to(device)
        self.std = self.std.to(device)
        self.cov = self.cov.to(device)
        self.mean = self.mean.to(device)
        return self
        
    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        normal = torch.distributions.Normal(0.0, 1.0)
        # Sample und Transformation (z @ A.T)
        z = normal.sample((n, self.dim))
        x = z @ self.A.T
        return x.to(device=device, dtype=dtype)

    def log_prob(self, x: Tensor) -> Tensor:
        device, dtype = _as_device_dtype(x.device, x.dtype)
        mvn = torch.distributions.MultivariateNormal(
            loc=self.mean.to(device, dtype), 
            covariance_matrix=self.cov.to(device, dtype)
        )
        return mvn.log_prob(x)


class MSGM_Cauchy(BaseDistribution2D):
    has_log_prob: bool = True

    def __init__(self, dim: int = 4, correlation: bool = True, normalized: bool = False, seed: int = 0):
        self.dim = dim
        self.scale = 1.0 / 50.0
        
        # Gleiches Seed-Konzept wie beim Gaussian für strikte Reproduzierbarkeit
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        
        if correlation:
            self.A = torch.randn(dim, dim)
        else:
            self.A = torch.eye(dim)
            
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        
        if normalized:
            self.A = torch.diag(1.0 / self.std) @ self.A 
            
        torch.set_rng_state(rng_state)
        
        # Für log_prob (Determinante der Transformationsmatrix)
        self.log_abs_det_A = torch.log(torch.abs(torch.det(self.A)))
        self.A_inv_T = torch.inverse(self.A).T

    def to(self, device):
        device = torch.device(device)
        self.A = self.A.to(device)
        self.std = self.std.to(device)
        self.log_abs_det_A = self.log_abs_det_A.to(device)
        self.A_inv_T = self.A_inv_T.to(device)
        return self

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        cauchy = torch.distributions.Cauchy(0.0, self.scale)
        z = cauchy.sample((n, self.dim))
        x = z @ self.A.T
        return x.to(device=device, dtype=dtype)

    def log_prob(self, x: Tensor) -> Tensor:
        # Transformation zurück in den unkorrelierten Z-Raum
        device, dtype = _as_device_dtype(x.device, x.dtype)
        A_inv_T = self.A_inv_T.to(device, dtype)
        z = x @ A_inv_T
        
        # Log-Prob von unabhängigen Cauchy-Variablen
        cauchy = torch.distributions.Cauchy(0.0, self.scale)
        log_prob_z = cauchy.log_prob(z).sum(dim=-1)
        
        # Change of variables Korrektur
        log_prob_x = log_prob_z - self.log_abs_det_A.to(device, dtype)
        return log_prob_x
    
class MSGM_PIV(BaseDistribution2D):
    has_log_prob: bool = False

    def __init__(
        self, 
        dim: int = 1024, 
        normalized: bool = False
    ):
        self.normalized = normalized
        self.name = 'PIV'
        
        # Pfad zum "largerImage" Ordner anpassen!
        self.data_dir = Path(__file__).parent.parent.parent / "MSGM-submission-main" / "data" / "largerImage"
            
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Daten-Ordner nicht gefunden: {self.data_dir}")

        print(f"Loading authors' raw PIV data from: {self.data_dir}")

        # 1. Daten laden und stapeln -> (Samples, 8192)
        files = sorted(self.data_dir.glob("*_vortdiv.npy"))
        if not files:
            raise FileNotFoundError(f"Keine *_vortdiv.npy Dateien in {self.data_dir} gefunden!")
        raw_data = np.vstack([np.load(f) for f in files])

        # 2. Skalieren und Zentrieren
        raw_data = raw_data / 2.5
        raw_data = raw_data - raw_data.mean(axis=0)

        # 3. Reshape, Smoothing (2) und Subsampling auf 32x32
        npixelx_max = 64
        npixelx = int(np.sqrt(dim)) # 32
        
        # Umformen auf (Samples, 64, 64, 2) und Kanal 0 (Vorticity) extrahieren
        raw_data = raw_data.reshape((raw_data.shape[0], npixelx_max, npixelx_max, 2), order='F')
        raw_data = raw_data[:, :, :, 0]

        # Gauß-Filter (smoothing=2 Logik der Autoren)
        sigmax = npixelx_max // npixelx  # 64 // 32 = 2
        raw_data *= 4.0
        for i in range(raw_data.shape[0]):
            raw_data[i, :, :] = gaussian_filter(raw_data[i, :, :], sigma=sigmax)

        # Räumliches Subsampling
        ix = np.linspace(0, raw_data.shape[1]-1, npixelx, dtype=int)
        iy = np.linspace(0, raw_data.shape[2]-1, npixelx, dtype=int)
        raw_data = raw_data[:, ix, :]
        raw_data = raw_data[:, :, iy]

        # Flachklopfen auf Ziel-Dimension 1024
        raw_data = raw_data.reshape((raw_data.shape[0], dim), order='F')

        self.dim = raw_data.shape[1]

        # 4. Train / Test Split (1/3 für Test)
        n_test = raw_data.shape[0] // 3
        train_data = raw_data[0:-n_test, :]
        test_data = raw_data[-n_test:, :]
        
        # 5. Standardabweichung auf gesamtem Dataset (wie Autoren)
        self.std = raw_data.std(axis=0)

        # Normalisieren
        if normalized:
            train_data = train_data / (self.std + 1e-8)
            test_data = test_data / (self.std + 1e-8)
            self.name += '_norm'

        # 6. Als PyTorch Tensoren im RAM speichern
        self.data = torch.from_numpy(train_data).to(torch.float32)
        self.data_test = torch.from_numpy(test_data).to(torch.float32)
        self.std_tensor = torch.from_numpy(self.std).to(torch.float32)

    def to(self, device):
        device = torch.device(device)
        self.data = self.data.to(device)
        self.data_test = self.data_test.to(device)
        self.std_tensor = self.std_tensor.to(device)
        return self

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data.shape[0], (n,))
        x = self.data[idx]
        return x.to(device=device, dtype=dtype)
        
    def sampletest(
        self, 
        n: int, 
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data_test.shape[0], (n,))
        x = self.data_test[idx]
        return x.to(device=device, dtype=dtype)

def get_distribution(name: str, **kwargs):
    name = name.lower()
    if name in {"msgm_swissroll", "msgm-swissroll"}:
        return MSGM_SwissRoll(**kwargs)
    if name in {"msgm_gaussian", "msgm-gaussian"}:
        return MSGM_Gaussian(**kwargs)
    if name in {"msgm_cauchy", "msgm-cauchy"}:
        return MSGM_Cauchy(**kwargs)
    if name in {"msgm_piv", "msgm-piv"}:
        return MSGM_PIV(**kwargs)
    if name in {"thinangles", "thin-angles", "thin-arc"}:
        return ThinAngles(**kwargs)
    if name in {"radialpareto", "radial-pareto"}:
        return RadialPareto(**kwargs)
    if name in {"isotropicheavytail", "isotropic-heavy-tail", "studentt"}:
        return IsotropicHeavyTail(**kwargs)
    if name in {"weirdgmm", "weird-gmm"}:
        return WeirdGMM(**kwargs)
    if name in {"rareoutliergmm", "rare-outlier-gmm"}:
        return RareOutlierGMM(**kwargs)
    if name in {"ringgmm", "ring-gmm"}:
        return RingGMM(**kwargs)
    if name in {"tworings", "two-rings", "concentricrings"}:
        return TwoRings(**kwargs)
    if name in {"twospirals", "two-spirals"}:
        return TwoSpirals(**kwargs)
    if name in {"thincross", "thin-cross", "crux"}:
        return ThinCross(**kwargs)
    if name in {"quantile_only"}:
        return QuantileOnlySampler(**kwargs)
    if name in {"anisotropicgauss"}:
        return AnisotropicGaussianSampler(**kwargs)
    if name in {"unigauss"}:
        return SimpleGaussianSampler(**kwargs)
    if name in {"spiral"}:
        return SpiralGenerator(**kwargs)
    if name in {"snail"}:
        return SnailDistribution(**kwargs)
    if name in {"donut"}:
        return Donut(**kwargs)
    if name in {"multigauss"}:
        return MultiGaussianSampler(**kwargs)
    if name in {"checker", "checkerboard"}:
        return CheckerboardStripes(**kwargs)
    if name in {"gridgmm", "gridgmm9", "gmmgrid"}:
        return GridGMM9(**kwargs)
    if name in {"funnel", "nealfunnel"}:
        base = NealFunnel2D(**kwargs)
        mean = torch.zeros(2)
        std = torch.tensor(
            [
                base.sigma1,
                math.exp(0.25 * (base.alpha ** 2) * (base.sigma1 ** 2)),
            ]
        )
        return ZScoreWrapper(base, mean, std)
    if name in {"mnist"}:
        return MNISTSampler(**kwargs)
    if name in {"cifar", "cifar10", "cifar-10"}:
        return CIFAR10Sampler(**kwargs)
    raise ValueError(f"Unknown distribution name: {name}")
