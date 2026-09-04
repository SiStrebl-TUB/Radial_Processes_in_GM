
import json
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import ot  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ot = None

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def seed_all(seed: int) -> None:
    """Seed python, numpy, and torch (including CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fixed_sampler(sampler, *, seed: int, device: torch.device) -> Callable[[int, int], torch.Tensor]:
    """Return a sampler that always yields the same batch for a given seed offset."""
    def _sample(batch_size: int, seed_offset: int = 0) -> torch.Tensor:
        devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + seed_offset)
            if device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + seed_offset)
            return sampler.sample(batch_size, device=device, dtype=torch.float32)
    return _sample


def make_fixed_uniform(shape: Tuple[int, ...], *, seed: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a fixed tensor of uniform samples in [0, 1) with deterministic seeding."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    base = torch.rand(shape, generator=gen, dtype=dtype)
    return base.to(device)


def count_parameters(module: torch.nn.Module) -> int:
    """Return the total number of parameters in the module."""
    return sum(int(p.numel()) for p in module.parameters())


def write_model_size_summary(runs_dir: str, stats: Dict[str, object]) -> Path:
    """Persist model size statistics to `<runs_dir>/model_size.json`."""
    path = Path(runs_dir) / "model_size.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in stats.items():
        if isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, sort_keys=True)
    return path


def ensure_args_defaults(args, image_shape):
    """Populate optional training flags with defaults if missing."""
    defaults = {
        "latent_viz_samples": 0,
        "latent_atlas_grid": 1,
        "sample_vis_interval": 0,
        "sample_vis_count": 0,
        "sample_vis_nrow": 8,
        "fid_eval_interval": 0,
        "fid_num_gen": 0,
        "fid_batch_size": getattr(args, "batch_size", 1),
        "fid_gen_batch": getattr(args, "batch_size", 1),
        "fid_image_size": image_shape[-1] if image_shape is not None else 0,
        "q_loss_weight": 1.0,
        "q_u_eps": 0.0,
        "lambda_reg": 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


@dataclass
class Paths:
    sample_dir: str
    checkpoint_dir: str


def prepare_paths(args) -> Paths:
    """Ensure run directories exist and return their locations."""
    sample_dir = os.path.join(args.runs_dir, "samples")
    checkpoint_dir = os.path.join(args.runs_dir, "quantile_fm")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return Paths(sample_dir, checkpoint_dir)


def minibatch_ot_pairing(x0, x1, *, entropic_eps=None, hard_match=True):
    """
    x0: (B, D) source batch ~ q0
    x1: (B, D) target batch ~ q1
    entropic_eps: ignored (always uses exact EMD as requested)
    hard_match:   ignored (pairing via row-wise argmax of EMD plan)

    Returns:
        idx1: (B,) indices so that pair i is (x0[i], x1[idx1[i]])
        P   : (B,B) EMD transport plan (torch.Tensor)
    """
    if ot is None:
        raise ImportError("POT (Python Optimal Transport) package is required for OT pairing")

    if x0.shape[0] != x1.shape[0]:
        raise ValueError(f"x0 and x1 must have same batch size; got {x0.shape[0]} vs {x1.shape[0]}")

    device = x0.device
    with torch.no_grad():
        C= torch.cdist(x0,x1).cpu().numpy()
        # CPU/double for POT
        x0d = x0.detach().cpu().numpy()
        x1d = x1.detach().cpu().numpy()

        # Squared Euclidean cost
        #C = ot.dist(x0d, x1d, metric='euclidean') ** 2
        a = ot.unif(len(x0d))
        b = ot.unif(len(x1d))

        # Exact EMD plan
        P_np = ot.emd(a, b, C)

        # Back to torch on original device
        P = torch.tensor(P_np, dtype=torch.float32, device=device)

        # Row-wise argmax pairing (ties resolve to first max)
        idx1 = torch.argmax(P, dim=0)

    return idx1, P

def spherical_ot_pairing(set_a: torch.Tensor, set_b: torch.Tensor):
    """
    Finds the optimal permutation of set_b to align with set_a such that
    the total spherical distance between pairs is minimized.
    
    Args:
        set_a: (B, D) Tensor of unit vectors.
        set_b: (B, D) Tensor of unit vectors.
        
    Returns:
        indices: (B,) LongTensor. Use this to reorder set_b.
    """
    # Ensure inputs are on CPU for SciPy, but keep track of original device
    device = set_a.device
    batch_size = set_a.shape[0]
    a_cpu = set_a.detach().cpu()
    b_cpu = set_b.detach().cpu()
    
    # 1. Compute Similarity Matrix (Dot Product)
    # We want to maximize dot product sum <=> minimize negative dot product sum.
    # Matrix shape: (B, B) where entry [i, j] is similarity between a[i] and b[j]
    similarity_matrix = torch.mm(a_cpu, b_cpu.t())
    
    # 2. Convert to Cost Matrix
    # We negate similarity because linear_sum_assignment finds the MINIMUM cost
    cost_matrix = -similarity_matrix.numpy()
    
    # 3. Solve Linear Sum Assignment (Hungarian Algorithm)
    # row_idx will be [0, 1, 2, ...], col_idx will be the permutation we want
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    # 4. Create Transport Plan (Permutation Matrix)
    # We create a matrix of zeros and place 1s at the optimal (row, col) positions
    plan = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
    plan[row_idx, col_idx] = 1.0
    # 4. Convert back to Tensor
    permuted_indices = torch.tensor(col_idx, dtype=torch.long, device=device)
    
    return permuted_indices, plan

def max_sliced_ot_pairing(
    x: torch.Tensor, 
    y: torch.Tensor, 
    num_iterations: int = 15, 
    lr: float = 0.5
) -> torch.Tensor:
    """
    Findet die optimale 1D-Projektionsachse zur Trennung der Verteilungen und 
    gibt die entsprechenden Coupling-Indizes zurück.
    
    Args:
        x: Source-Batch (z. B. Target-Daten auf der Sphäre) der Form (B, d)
        y: Target-Batch (z. B. Noise-Daten auf der Sphäre) der Form (B, d)
        num_iterations: Anzahl der Gradient-Ascent Schritte für die Achse
        lr: Lernrate für die Achsen-Optimierung
    """
    B, d = x.shape
    device = x.device

    # Initialisiere zufällige Projektionsachse auf der Einheitssphäre
    theta = torch.randn(d, 1, device=device)
    theta = F.normalize(theta, p=2, dim=0)
    theta.requires_grad_(True)

    # Wir nutzen SGD, da wir nur diesen einen Vektor updaten
    optimizer = torch.optim.SGD([theta], lr=lr)

    # Gradient Ascent, um die W2-Distanz auf der Achse zu maximieren
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        # 1D-Projektionen
        u = torch.matmul(x, theta).squeeze(1)
        v = torch.matmul(y, theta).squeeze(1)
        
        # Sortieren
        u_sorted, _ = torch.sort(u)
        v_sorted, _ = torch.sort(v)
        
        # W2-Distanz berechnen (negativ für Maximierung via Gradient Descent)
        loss = -torch.mean((u_sorted - v_sorted) ** 2)
        
        loss.backward()
        optimizer.step()
        
        # Achse nach jedem Schritt wieder auf die Sphäre normieren
        with torch.no_grad():
            theta.copy_(F.normalize(theta, p=2, dim=0))

    # Finales Coupling mit der optimierten Achse (ohne Gradienten)
    with torch.no_grad():
        u_final = torch.matmul(x, theta).squeeze(1)
        v_final = torch.matmul(y, theta).squeeze(1)
        
        _, idx_x = torch.sort(u_final)
        _, idx_y = torch.sort(v_final)
        
        # Zuweisungs-Array konstruieren: x[idx] wird y[idx_y] zugeordnet
        idx_best = torch.empty(B, dtype=torch.long, device=device)
        idx_best[idx_x] = idx_y
        
    return idx_best

def sliced_ot_pairing(targets, noise):
    """
    targets, noise: (B, D) tensors
    Beide sollten für die Projektion idealerweise normiert sein.
    """
    B, D = targets.shape
    device = targets.device
    
    # 1. Zufällige globale Richtung generieren
    v = torch.randn(D, device=device)
    v = torch.nn.functional.normalize(v, dim=0)
    
    # 2. Beide Sets auf diese Achse projizieren (1D Werte)
    proj_targets = torch.mv(targets, v)
    proj_noise = torch.mv(noise, v)
    
    # 3. Sortieren und Indizes extrahieren
    _, idx_targets = torch.sort(proj_targets)
    _, idx_noise = torch.sort(proj_noise)
    
    # (Optional) Wenn du nur die Indizes zurückgeben willst, um
    # danach die unnormierten Radii zuzuordnen:
    final_indices = torch.empty_like(idx_noise)
    final_indices[idx_targets] = idx_noise
    
    return final_indices

def oversample_heavy_tails(x_data, top_fraction=0.1):
    """
    Nimmt die Top X% der Daten (basierend auf der Norm) und kopiert sie so oft,
    dass sich die Batch-Size verdoppelt.
    
    Args:
        x_data: (B, D) Tensor
        top_fraction: Welcher Anteil gilt als "Outlier"? (z.B. 0.1 für Top 10%)
    
    Returns:
        x_augmented: (2*B, D) Tensor
    """
    B = x_data.shape[0]
    num_outliers = int(B * top_fraction)
    
    if num_outliers == 0:
        return torch.cat([x_data, x_data], dim=0)

    # 1. Normen berechnen
    norms = torch.linalg.vector_norm(x_data, dim=1)
    
    # 2. Die Indizes der größten Normen finden (Top k)
    _, top_indices = torch.topk(norms, k=num_outliers)
    
    # 3. Die Outlier-Samples extrahieren
    outliers = x_data[top_indices] # Shape (num_outliers, D)
    
    # 4. Auffüllen: Wir brauchen B neue Samples.
    # Wir wiederholen die Outliers so oft wie nötig.
    repeats = B // num_outliers
    remainder = B % num_outliers
    
    outliers_repeated = outliers.repeat(repeats, 1)
    
    # Falls es nicht glatt aufgeht, füllen wir den Rest mit den ersten Outliers auf
    if remainder > 0:
        outliers_repeated = torch.cat([outliers_repeated, outliers[:remainder]], dim=0)
        
    # 5. Zusammenfügen: Original-Daten + Vervielfachte Outliers
    x_augmented = torch.cat([x_data, outliers_repeated], dim=0)
    
    return x_augmented