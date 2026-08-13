import torch
import torch.nn as nn

class DummyBaseSDE:
    """
    Dummy-Klasse, um Kompatibilität mit dem MSGM-Auswertungscode zu gewährleisten.
    Das Hauptskript greift für die Namensgebung von Plot-Dateien und für einige Logs 
    auf Attribute von 'base_sde' zu. Diese fangen wir hier sicher ab.
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
        
        # WICHTIG: Der RK4-Solver (rk4_stratonovich_sampler) fragt intern 
        # sde.T.device und sde.T.item() ab. T muss daher zwingend ein Tensor sein!
        self.T = torch.tensor([1.0], device=device)
        self.base_sde = DummyBaseSDE()
        
    def mu_Strato(self, t, x, lmbd=0.):
        """
        Dies ist der Drift-Term (das deterministische Vektorfeld), 
        das vom RK4-Solver aufgerufen wird.
        
        ZEITACHSE ERKLÄRT:
        - Der Solver startet bei t=0 (Beginn der Generierung = Rauschen) 
          und integriert schrittweise hoch bis t=T (Ende = Daten).
        - Dein Flow-Matching-Modell wurde so trainiert, dass t=1 das Rauschen 
          und t=0 die echten Daten sind.
        - Wir rechnen also um: t_phys = 1.0 - (t / T). 
          (Wenn der Solver bei 0 startet, sieht dein Modell eine 1).
          
        MINUSZEICHEN ERKLÄRT:
        - Da wir in der physikalischen Zeit "rückwärts" gehen (von 1 nach 0), 
          müssen wir das Vektorfeld deines Modells negieren, damit der Solver 
          in die richtige Richtung integriert.
        """
        # 1. Zeitachse umkehren und normieren
        t_phys = 1.0 - (t / self.T)
        
        # 2. Modell aufrufen (Reihenfolge t, x passt zu deinen Klassen)
        v_raw = self.fm_model(t_phys, x)
        
        # 3. Vektorfeld für die Rückwärtsintegration negieren
        return -1.0 * v_raw

    def mu(self, t, x, lmbd=0.):
        """
        Fallback-Methode. Wird genutzt, falls im Hauptskript anstelle des RK4
        versehentlich der Euler-Maruyama-Sampler (EMstep) aufgerufen wird.
        """
        return self.mu_Strato(t, x, lmbd)

    def sigma(self, t, x, lmbd=0.):
        """
        Der Diffusions-Term (das Rauschen).
        Da Flow Matching auf einer deterministischen Probability Flow ODE basiert,
        gibt es während der Generierung kein Rauschen. 
        Wir geben daher einfach einen Tensor aus Nullen zurück.
        """
        return torch.zeros_like(x)
    
    def latent_sample(self, num_samples, dim):
        # 1. Erzeuge Einheitsvektoren (Normalverteilung teilen durch deren Norm)
        noise = torch.randn(num_samples, dim, device=self.T.device)
        unit_vectors = noise / (noise.norm(dim=1, keepdim=True) + 1e-8)
        
        sampled_target_norm = self.sampler.sample(num_samples).norm(dim=1, keepdim=True)
        
        return unit_vectors * sampled_target_norm