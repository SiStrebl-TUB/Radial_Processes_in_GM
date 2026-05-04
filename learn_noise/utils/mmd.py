import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

def memory_efficient_mmd(x, y, bandwidths=[0.1, 1.0, 10.0], chunk_size=1000, log_to_wandb=True, step=None, target_name="Dataset"):
    """
    Berechnet die MMD speichereffizient in Blöcken (Chunks), um OOM-Errors 
    bei großen N (z.B. 50.000) zu vermeiden.
    Optional: Generiert einen Side-by-Side Plot der Inputs und loggt ihn in W&B.
    """
    
    # --- VISUALIZATION BLOCK ---
    if log_to_wandb:
        # Safely move tensors to CPU and convert to NumPy for matplotlib
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
        
        # Create side-by-side subplots with shared axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        
        # Calculate global limits to ensure both plots have the exact same scale
        x_min = min(x_np[:, 0].min(), y_np[:, 0].min())
        x_max = max(x_np[:, 0].max(), y_np[:, 0].max())
        y_min = min(x_np[:, 1].min(), y_np[:, 1].min())
        y_max = max(x_np[:, 1].max(), y_np[:, 1].max())
        
        # Add a 5% margin to the borders
        margin_x = (x_max - x_min) * 0.05
        margin_y = (y_max - y_min) * 0.05
        
        ax1.set_xlim(x_min - margin_x, x_max + margin_x)
        ax1.set_ylim(y_min - margin_y, y_max + margin_y)

        # Plot y (Ground Truth) on the left
        ax1.scatter(y_np[:, 0], y_np[:, 1], alpha=0.5, color='blue', s=5)
        ax1.set_title("Target / Ground Truth (y)")
        ax1.grid(True, alpha=0.3)

        # Plot x (Generated Data) on the right
        ax2.scatter(x_np[:, 0], x_np[:, 1], alpha=0.5, color='red', s=5)
        ax2.set_title("Generated Data (x)")
        ax2.grid(True, alpha=0.3)

        # Main Title
        step_str = f" | Step: {step}" if step is not None else ""
        fig.suptitle(f"MMD Input Comparison: {target_name}{step_str}", fontsize=14)
        plt.tight_layout()

        # Log to Weights & Biases
        log_key = f"eval/mmd_inputs_{target_name}"
        if step is not None:
            wandb.log({log_key: wandb.Image(fig)}, step=step)
        else:
            wandb.log({log_key: wandb.Image(fig)})
            
        # CRITICAL: Close the figure to prevent RAM leaks!
        plt.close(fig)
    # ---------------------------

    N = x.shape[0]
    M = y.shape[0]
    
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    
    # 1. XX Summe berechnen
    for i in range(0, N, chunk_size):
        x_chunk = x[i:i+chunk_size]
        dxx = torch.cdist(x_chunk, x, p=2).pow(2)
        for a in bandwidths:
            sum_xx += torch.exp(-0.5 * dxx / a).sum().item()
            
    # 2. YY Summe berechnen
    for i in range(0, M, chunk_size):
        y_chunk = y[i:i+chunk_size]
        dyy = torch.cdist(y_chunk, y, p=2).pow(2)
        for a in bandwidths:
            sum_yy += torch.exp(-0.5 * dyy / a).sum().item()
            
    # 3. XY Summe berechnen
    for i in range(0, N, chunk_size):
        x_chunk = x[i:i+chunk_size]
        dxy = torch.cdist(x_chunk, y, p=2).pow(2)
        for a in bandwidths:
            sum_xy += torch.exp(-0.5 * dxy / a).sum().item()
            
    # MMD Formel zusammenbauen
    mmd_sq = (sum_xx / (N * N)) + (sum_yy / (M * M)) - 2.0 * (sum_xy / (N * M))
    
    return mmd_sq