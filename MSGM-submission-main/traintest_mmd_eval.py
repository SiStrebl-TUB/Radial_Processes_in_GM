import torch
import numpy as np
import random
import gc

# Importiere deine PIV Klasse (passe den Import an, falls er bei dir anders heißt)
from data import PIV 

# Device Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Die speicher- und präzisionsoptimierten MMD-Funktionen
@torch.no_grad()
def compute_kernel(x, y):
    with torch.no_grad():
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        dim = float(x.size(1))
        
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
        mmd = x_kernel.mean() + y_kernel.mean() - 2.0 * xy_kernel.mean()
        del x_kernel, y_kernel, xy_kernel
        gc.collect()
    return mmd

def main():
    num_samples = 1000  # Feste Größe von 1000 Samples
    num_runs = 25       # Wie oft wollen wir das mit verschiedenen Seeds testen?
    dim = 1024          # Entspricht 32x32 beim PIV
    
    results = []

    print(f"Starte Varianz-Analyse mit {num_runs} Durchläufen (jeweils N = {num_samples})...\n")

    for run in range(num_runs):
        # Wir nutzen für jeden Run einen völlig anderen Seed, 
        # um unterschiedliche Kombinationen aus dem "Heuhaufen" zu ziehen.
        seed = run * 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Sampler initialisieren
        sampler = PIV(dim, normalized=False, largeImage=True, smoothing=2, localized=False, few_data=False)
        
        # Testdaten ziehen
        xtest = sampler.sampletest(2500).to(device) # Ziehen aus dem vollen Pool
        xtest_sub = xtest[0:num_samples-1, :]      # Auf 1000 (bzw. 999) begrenzen
        
        std_norm = torch.ones((xtest.shape[1]))
        std_norm_t = torch.as_tensor(std_norm, device=device, dtype=torch.float64)

        # Trainingsdaten ziehen
        n_sub = xtest_sub.shape[0]
        x_train = sampler.sample(n_sub).to(device)

        # MMD berechnen
        dist_train_to_test = compute_mmd(std_norm_t * x_train, std_norm_t * xtest_sub)
        mmd_value = dist_train_to_test.sqrt().item()
        
        results.append(mmd_value)
        print(f"Run {run+1:2d} (Seed {seed:3d}): MMD = {mmd_value:.6f}")

    # Statistik ausgeben
    results = np.array(results)
    print("\n" + "#"*40)
    print("### STATISTISCHE AUSWERTUNG (N = 1000) ###")
    print("#"*40)
    print(f"Mittelwert (Mean): {results.mean():.6f}")
    print(f"Standardabweichung: {results.std():.6f}")
    print(f"Minimum:          {results.min():.6f}")
    print(f"Maximum:          {results.max():.6f}")
    print("-" * 40)

if __name__ == '__main__':
    main()