import math
from typing import Optional

import torch
import torch.nn as nn

import time as _time


class TorchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._labels: Optional[torch.Tensor] = None

    def forward(self,x,t):
       
        time = t.repeat(x.shape[0])[:, None].to(device=x.device, dtype=x.dtype)
        labels = self._labels
        if labels is not None:
            if labels.shape[0] != x.shape[0]:
                raise ValueError("Mismatch between stored labels and batch size in TorchWrapper")
            out = self.model(time, x, labels)
        else:
            out = self.model(time, x)
        return out

    def set_labels(self, labels: Optional[torch.Tensor]) -> None:
        self._labels = labels


class VelocityFieldAdapter(nn.Module):
    """Adapts image-space models (e.g. UNets) to the (t, x[, labels]) API."""

    def __init__(self, model: nn.Module, image_shape):
        super().__init__()
        self.model = model
        self.image_shape = tuple(image_shape)
        self._flat_dim = math.prod(self.image_shape)

    def forward(self, t: torch.Tensor, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, *self.image_shape)
        timesteps = t.view(batch).to(x_reshaped.device)
        if labels is not None:
            labels = labels.to(device=x_reshaped.device, dtype=torch.long)
            out = self.model(x_reshaped, timesteps, y=labels)
        else:
            out = self.model(x_reshaped, timesteps)
        return out.reshape(batch, self._flat_dim)
    
class ODEWrapper(torch.nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.fmap = fmap
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.fmap(x, t)


class PotentialGradWrapper(nn.Module):
    """
    Wraps a scalar potential s(t, x) and exposes its spatial gradient ∇_x s(t, x)
    so it can be used as an ODE drift field.
    Expects `potential` to implement forward(time, x) -> (B, 1).
    """
    def __init__(self, potential: nn.Module):
        super().__init__()
        self.potential = potential

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure proper shapes
        if t.ndim == 0:
            t_in = t.repeat(x.shape[0])[:, None].float()
        else:
            t_in = t

        # Compute gradient w.r.t. x only; do not build large graphs
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            s = self.potential(t_in, x_req)  # (B, 1)
            grad_x = torch.autograd.grad(s.sum(), x_req, create_graph=False, retain_graph=False)[0]
        return grad_x
    
class SphericalProjectedModel(nn.Module):
    """
    Wraps a base model and explicitly projects the output velocity field
    onto the tangent space of the sphere at the current position x.
    
    This ensures mathematically that <x, v> = 0, meaning the velocity
    has NO radial component that would push the particle off the surface.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # 1. Vorhersage des Basis-Modells (kann Radial-Drift enthalten)
        v_raw = self.model(t, x, *args, **kwargs)
        
        # 2. Berechnung des Normalenvektors an Position x
        # Auf einer Sphäre ist der Normalenvektor einfach die Richtung vom Ursprung zu x.
        # Wir normalisieren x, um einen sauberen Richtungsvektor n zu haben.
        x_norm_val = torch.norm(x, dim=1, keepdim=True) + 1e-8
        n = x / x_norm_val
        
        # 3. Projektion auf den Tangentialraum
        # Formel: v_tangent = v_raw - (v_raw . n) * n
        # (v_raw . n) ist der Anteil der Geschwindigkeit, der nach außen/innen zeigt.
        
        radial_component = (v_raw * n).sum(dim=1, keepdim=True)
        # Wir ziehen den radialen Anteil ab.
        v_tangent = v_raw - (radial_component * n)     
        return v_tangent

class AngularVelocityWrapper(nn.Module):
    """
    Konvertiert die Ausgabe des Modells (Winkelgeschwindigkeit)
    zurück in lineare Geschwindigkeit für den Solver.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # 1. Modell fragen (gibt rad/s zurück)
        v_log1p = self.model(t, x, *args, **kwargs)
        
        # 2. Aktuellen Radius bestimmen
        radius = x.norm(dim=1, keepdim=True)
        log1p_radius = torch.log1p(radius)
        factor = 1#radius/(log1p_radius+1e-8)  # Vermeidet Division durch Null
        # 3. Rückrechnen: v_linear = v_angular * Radius
        v_linear = v_log1p * factor
        
        return v_linear
    
class NormalizedRadiusConditionedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # 1. Radius retten (bevor wir ihn weg-normalisieren!)
        radius = x.norm(dim=1, keepdim=True)  # (B, 1)
        x_normal = x / (radius + 1e-8)  # Vermeidet Division durch Null
        x_with_rad = torch.cat([x_normal, radius], dim=1)  # (B, D+1)
        return self.model(t, x_with_rad, *args, **kwargs) * radius
    
class MinimalRadiusConcatWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        # 1. Radius berechnen
        radius = x.norm(dim=1, keepdim=True) + 1e-8
        
        # 2. Richtung isolieren (Numerisch stabil [-1, 1])
        x_dir = x / radius
        
        # 3. Radius-Info komprimieren (0 bis ~3.0)
        r_feature = torch.log1p(radius)
        
        # 4. Zusammenfügen: Aus 2D wird 3D
        net_input = torch.cat([x_dir, r_feature], dim=1)
        
        return self.model(t, net_input, *args, **kwargs)
        
class AngularInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        
        x_norm = x.norm(dim=1, keepdim=True) + 1e-8
        scale = torch.log1p(x_norm) / x_norm 
        x = x * scale
        #x_lograd = x / x_norm * torch.log1p(x_norm)
        velocity_shortened = self.model(t, x, *args, **kwargs)
        return velocity_shortened / (scale + 1e-8) #* x_norm  # Rückskalierung auf lineare Geschwindigkeit

class LogFourierMLPWrapper(nn.Module):
    def __init__(self, model, num_bands=4, max_log_radius=12.0):
        super().__init__()
        self.model = model
        
        # max_log_radius = 12.0 entspricht einem echten Radius von ca. 160.000 !
        # Damit bist du auf der sicheren Seite für quasi alles.
        
        # Frequenzen im LOG-RAUM berechnen
        # Wir decken den Bereich [0, 12] ab.
        freqs = 2.0 ** torch.arange(num_bands) * torch.pi / max_log_radius
        self.register_buffer('freqs', freqs)

    def forward(self, t, x, *args, **kwargs):
        # 1. Radius berechnen
        radius = x.norm(dim=1, keepdim=True)
        
        # 2. Richtung normalisieren (Safe Division)
        x_dir = x / (radius + 1e-8)

        # 3. Ab in den Log-Raum (Das ist der Stauch-Trick)
        log_r = torch.log1p(radius) # Aus 700 wird 6.55
        
        # 4. Fourier Features auf dem LOG-Wert berechnen
        # Das gibt dem MLP die "Textur" und Genauigkeit
        fourier_features = torch.cat([
            torch.sin(log_r * self.freqs),
            torch.cos(log_r * self.freqs)
        ], dim=-1)

        # 5. WICHTIG: Den rohen log_r auch mitgeben! (Monotonie)
        # Hilft dem MLP bei der Extrapolation für Werte > Training-Set
        # Input-Vektor: [Dir(2), RawLog(1), Sin(4), Cos(4)] -> 11 Dimensionen
        net_input = torch.cat([x_dir, log_r, fourier_features], dim=1)
        
        return self.model(t, net_input, *args, **kwargs)