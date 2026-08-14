import numpy as np
import torch
import torch.nn as nn


class DummyBaseSDE:
    """
    Dummy-Klasse, um Kompatibilität mit dem MSGM-Auswertungscode zu gewährleisten.
    Das Hauptskript greift für die Namensgebung von Plot-Dateien und für einige Logs 
    auf Attribute von 'base_sde' zu.
    """
    def __init__(self):
        self.name_SDE = "FlowMatching"
        self.t_epsilon = 0.001 
        self.num_steps_forward = 100


class FlowMatchingWrapper(nn.Module):
    def __init__(self, fm_model, device, sampler):
        """
        Kapselt dein Flow-Matching-Modell so, dass es für den MSGM-Solver
        wie eine gewöhnliche SDE aussieht.
        """
        super().__init__()
        self.fm_model = fm_model
        self.sampler = sampler
        
        # Buffer zieht automatisch bei self.to(device) mit auf die GPU/CPU
        self.register_buffer("T", torch.tensor([1.0], device=device, dtype=torch.float32))
        self.base_sde = DummyBaseSDE()
        
    @property
    def device(self):
        return self.T.device

    def mu_Strato(self, t, x, lmbd=0.0):
        """
        Drift-Term (deterministisches Vektorfeld) für den RK4-Solver.
        """
        # 1. Sicherstellen, dass t denselben Typ und dasselbe Device wie x hat
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t_tensor = t.to(device=x.device, dtype=x.dtype)

        # 2. Zeitachse umkehren und normieren
        t_phys = 1.0 - (t_tensor / self.T.to(device=x.device, dtype=x.dtype))
        
        # 3. Modell aufrufen
        v_raw = self.fm_model(t_phys, x)
        
        # 4. Vektorfeld für die Rückwärtsintegration negieren
        return -1.0 * v_raw

    def mu(self, t, x, lmbd=0.0):
        """
        Fallback für Euler-Maruyama-Schritte.
        """
        return self.mu_Strato(t, x, lmbd)

    def sigma(self, t, x, lmbd=0.0):
        """
        Diffusions-Term (Null-Tensor auf exakt demselben Device/Dtype wie x).
        """
        return torch.zeros_like(x)
    
    def latent_sample(self, num_samples, dim):
        target_device = self.T.device
        
        # 1. Einheitsvektoren direkt auf dem Ziel-Device erzeugen
        noise = torch.randn(num_samples, dim, device=target_device)
        unit_vectors = noise / (noise.norm(dim=1, keepdim=True) + 1e-8)
        
        # 2. Target-Norm berechnen und sicher auf das Ziel-Device schieben
        raw_samples = self.sampler.sample(num_samples)
        if isinstance(raw_samples, np.ndarray):
            raw_samples = torch.from_numpy(raw_samples)
            
        sampled_target_norm = raw_samples.to(
            device=target_device, 
            dtype=unit_vectors.dtype
        ).norm(dim=1, keepdim=True)
        
        return unit_vectors * sampled_target_norm