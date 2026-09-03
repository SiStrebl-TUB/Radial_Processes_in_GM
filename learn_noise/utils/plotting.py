import wandb
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, NullLocator

from learn_noise.utils.colors import COL_DENSITY

def compute_log_joint_grid(
        sampler, 
        x1_range, 
        x2_range, 
        n1=300, 
        n2=300
    ):
    x1_lin = np.linspace(*x1_range, n1)
    x2_lin = np.linspace(*x2_range, n2)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing="ij")
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    logp = sampler.log_prob(torch.from_numpy(grid)).cpu().numpy().reshape(n1, n2)
    return x1_lin, x2_lin, logp

def _analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 80) -> np.ndarray:
    """
    Compute the marginal p(x2) via Gauss–Hermite quadrature:
      x1 ~ N(0, scale1^2), x2 | x1 ~ N(0, exp(x1))
      p(x2) = E_{x1}[ N(x2; 0, exp(x1)) ]

    Using GH: E_{Z~N(0,1)}[f(Z)] ≈ (1/sqrt(pi)) Σ w_i f(√2 x_i)
      => E_{X1~N(0,scale1^2)}[g(X1)] ≈ (1/sqrt(pi)) Σ w_i g(scale1 * √2 * x_i)

    Args:
      x2_grid: array of x2 values
      scale1:  std of x1 (e.g., 3.0)
      gh_n:    quadrature order
    """
    from numpy.polynomial.hermite import hermgauss

    x2 = x2_grid.astype(np.float64)
    nodes, weights = hermgauss(gh_n)
    # Transform nodes for Normal(0, scale1^2)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes  # shape (N,)
    w_norm = weights / np.sqrt(np.pi)          # weights for E[·]

    # For each x1, the conditional is N(0, var=exp(x1))
    var = np.exp(x1_vals)                      # (N,)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    # Compute mixture pdf across quadrature nodes
    # pdf_i(x2) = 1/sqrt(2π var_i) * exp(-x2^2 / (2 var_i))
    # Vectorized: (N,1) over x1 nodes vs (1,M) over x2 grid
    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(- (x2[None, :]**2) / (2.0 * var_col))
    pdf_matrix = coef * expo                    # (N, M)
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)

    # Numerical floor to avoid 0 on log-y plots
    return np.maximum(pdf, 1e-300)

def plot_funnel_tail_ccdf(args, sampler, generated_norms, step=0, filename="funnel_ccdf", key="tail_ccdf") -> None:
    """
    Erstellt einen Log-Log CCDF Plot für Neal's Funnel.
    
    generated_samples: 2D Numpy Array der generierten Daten (Shape: [N, 2])
    sigma1: Die Standardabweichung der x1-Komponente deines echten Funnels.
    """
    # 1. Radien der generierten Daten berechnen
    gen_radii = generated_norms
    sorted_gen = np.sort(gen_radii)
    n_gen = len(sorted_gen)
    empirical_ccdf = np.arange(n_gen, 0, -1) / n_gen

    # 2. "Theoretische" CCDF empirisch annähern
    # Wir generieren extrem viele echte Samples, um die Tails glatt darzustellen
    n_true = 10_000_000 
    target_samples = sampler.sample(n_true)# σ_2 = e^{x_1}
    true_x1 = target_samples[:, 0].detach().cpu().numpy()
    true_x2 = target_samples[:, 1].detach().cpu().numpy()
    true_radii = np.sqrt(true_x1**2 + true_x2**2)
    sorted_true = np.sort(true_radii)
    true_ccdf = np.arange(n_true, 0, -1) / n_true

    # 3. Plotten
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'

    # Empirische Daten (Generiert vom Modell)
    plt.plot(sorted_gen, empirical_ccdf, marker='.', linestyle='none', 
             color='#d62728', alpha=0.7, label='Generated Samples')
    
    # "Theoretische" Gerade (Wahres Neal's Funnel)
    plt.plot(sorted_true, true_ccdf, color='#1f77b4', linewidth=2.5, 
             linestyle='-', label=f"Target Neal's Funnel")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Radius $r=\|x\|$ (Log Scale)', fontsize=16)
    plt.ylabel(r'$\mathrm{Pr}(R > r)$ (Log Scale)', fontsize=16)
    plt.title("", fontsize=18)
    
    # Limits anpassen, damit der Plot nicht durch Ausreißer der 2 Mio echten Samples zerquetscht wird
    # Wir schneiden bei ca. 10^-4 oder 10^-5 ab, je nachdem wie viele generierte Samples du hast
    y_min_limit = max(1 / (n_gen * 10), 1e-6)
    plt.ylim(bottom=y_min_limit, top=1.5)
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # 4. Speichern & Loggen
    if args.runs_dir:
        out_path = os.path.join(args.runs_dir, f"{filename}_step_{step:06d}.png")
        fig.savefig(out_path, dpi=300)
        try:
            wandb.log({key: wandb.Image(out_path)}, step=step)
        except Exception:
            pass
            
    plt.close(fig)


def plot_pareto_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None
    ):
    
    # Convert tensors to numpy if they aren't already
    if torch.is_tensor(generated):
        generated = generated.detach().cpu().numpy()

    n_data = generated.shape[0]
    
    # Generate True Samples for the red baseline
    with torch.no_grad():
        S_data = sampler.sample(n_data).cpu().numpy()

    x1_d, x2_d = generated[:, 0], generated[:, 1]
    x1_m, x2_m = S_data[:, 0], S_data[:, 1]

    # ----- 1. Setup SymLog Bins and Grid -----
    # Linear threshold where the axis switches from linear to log
    # r_min is 2.0, so 5.0 is a great threshold to keep the core linear
    LIN_THRESH = 5.0  
    TAIL_MAX = 1000.0 # Extend axis to 10,000 to see the heavy tails
    
    # Create bins that are uniform in symlog space
    lin_pts = np.linspace(-LIN_THRESH, LIN_THRESH, 30)
    log_pos = np.geomspace(LIN_THRESH, TAIL_MAX, 40)
    log_neg = -np.geomspace(LIN_THRESH, TAIL_MAX, 40)[::-1]
    
    # This 'pts' array serves as both the meshgrid base and the histogram bins
    pts = np.concatenate([log_neg, lin_pts[1:-1], log_pos])
    
    # Evaluate analytic log-prob on this non-linear grid
    xx, yy = np.meshgrid(pts, pts)
    grid_tensor = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        logp_tensor = sampler.log_prob(grid_tensor)
        # Handle the -inf inside the core gracefully for plotting
        logp_tensor = torch.nan_to_num(logp_tensor, neginf=-50.0)
        logp = logp_tensor.view(xx.shape).cpu().numpy()

    # ----- 2. Figure Layout -----
    fig = plt.figure(figsize=(9, 9), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    # ----- 3. Main Axis: SymLog Scatter & Contour -----
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    
    # Because Pareto drops off heavily, we set a deep floor (-30 log prob)
    log_floor = -30.0 
    vmax = float(np.max(logp))
    
    # pcolormesh is required instead of imshow for non-linear grids
    ax_main.pcolormesh(
        xx, yy, logp, 
        cmap=cmap, 
        vmin=log_floor, 
        vmax=vmax, 
        shading='auto',
        zorder=0
    )

    # Scatter generated points
    ax_main.scatter(x1_d, x2_d, s=6, alpha=0.6,
                    color=teal, linewidths=0, edgecolors="none", rasterized=True, zorder=2)

    ax_main.set_xlabel(r"$x_1$", color="black")
    ax_main.set_ylabel(r"$x_2$", color="black")
    
    # Activate SymLog scale to squish the extreme tails into view
    ax_main.set_xscale('symlog', linthresh=LIN_THRESH)
    ax_main.set_yscale('symlog', linthresh=LIN_THRESH)
    ax_main.set_xlim(-TAIL_MAX, TAIL_MAX)
    ax_main.set_ylim(-TAIL_MAX, TAIL_MAX)

    # ----- 4. Top Axis: X Marginal (Heavy Tail view) -----
    # We use a log-scale on the Y-axis to see the power-law slope
    ax_top.set_yscale("log")
    
    # Plot histograms using our symlog-spaced bins so the widths look uniform
    ax_top.hist(x1_m, bins=pts, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x1_d, bins=pts, density=True, color=teal, alpha=0.35, edgecolor=teal)
    
    ax_top.tick_params(labelbottom=False)

    # ----- 5. Right Axis: Y Marginal (Heavy Tail view) -----
    # We use a log-scale on the X-axis for the probabilities
    ax_right.set_xscale("log")
    
    ax_right.hist(x2_d, bins=pts, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x2_m, bins=pts, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_2)$")

    # ----- 6. Export and Log -----
    if path is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'), bbox_inches='tight')
        
    if big_eval:
        wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)

    plt.close()

def plot_funnel_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None
    ):
    # ----- fixed axis ranges -----
    X2_MIN, X2_MAX = -999.0, 999.0   # horizontal (x-axis): x2
    X1_MIN, X1_MAX =  -20.0,    20.0   # vertical (y-axis):   x1

    n_data = generated.shape[0]
    # Use a larger true sample for a smoother red outline in tails
    n_true = n_data  # max(n_data, min(200_000, 10 * n_data))
    S_data = sampler.sample(n_true,)

    x1_d = generated[:, 0]
    x2_d = generated[:, 1]

    x1_m = S_data[:, 0]
    x2_m = S_data[:, 1]

    # ----- evaluate TRUE log p(x1,x2) on the fixed grid -----
    x1_lin, x2_lin, logp = compute_log_joint_grid(
        sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360
    )

    # histogram bins aligned with fixed axes
    bins_x2 = np.linspace(X2_MIN, X2_MAX, 50)
    bins_x1 = np.linspace(X1_MIN, X1_MAX, 50)

    # ----- figure -----
    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05 # <— small gap so top “1000” and right “0” don’t collide
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    # main: true log-joint with floor at -20 mapped to black
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    log_floor = -20.0
    vmax = float(np.max(logp))
    ax_main.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    # scatter (teal)
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5,
                    color=teal, linewidths=0, edgecolors="none", rasterized = True)

    ax_main.set_xlabel(r"$x_2$", color="white")
    ax_main.set_ylabel(r"$x_1$", color="white")
    # fixed limits
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)

    # top: x2 marginal (log-y) — data filled teal, model red outline
    ax_top.set_yscale("log")
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)

    # Analytic overlay for p(x2) using Gauss–Hermite quadrature
    x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
    scale1 = float(getattr(sampler, 'scale1', torch.tensor(3.0)).item())
    px2 = _analytic_funnel_x2_pdf(x2_centers, scale1=scale1, gh_n=80)
    ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95, label="analytic", zorder=5)
    ax_top.tick_params(labelbottom=False)

    # right: x1 marginal (horizontal) — data filled teal, model red outline
    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    out = "funnel_true_all_blackfloor_fixed.pdf"
    plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'), bbox_inches='tight')
    if big_eval:
      wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
      wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)

    plt.close()

"""def plot_pareto_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None
    ):
    Pareto 2D Plotting - MODIFIED: ONLY GROUND TRUTH (RED)
    
    # Convert tensors to numpy if they aren't already
    if torch.is_tensor(generated):
        generated = generated.detach().cpu().numpy()

    n_data = generated.shape[0]
    
    # Generate True Samples for the red baseline
    with torch.no_grad():
        S_data = sampler.sample(n_data).cpu().numpy()

    # Nur die echten Daten werden verwendet
    x1_m, x2_m = S_data[:, 0], S_data[:, 1]

    # ----- 1. Setup SymLog Bins and Grid -----
    LIN_THRESH = 5.0  
    TAIL_MAX = 1000.0 
    
    lin_pts = np.linspace(-LIN_THRESH, LIN_THRESH, 30)
    log_pos = np.geomspace(LIN_THRESH, TAIL_MAX, 40)
    log_neg = -np.geomspace(LIN_THRESH, TAIL_MAX, 40)[::-1]
    
    pts = np.concatenate([log_neg, lin_pts[1:-1], log_pos])
    
    xx, yy = np.meshgrid(pts, pts)
    grid_tensor = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        logp_tensor = sampler.log_prob(grid_tensor)
        logp_tensor = torch.nan_to_num(logp_tensor, neginf=-50.0)
        logp = logp_tensor.view(xx.shape).cpu().numpy()

    # ----- 2. Figure Layout -----
    fig = plt.figure(figsize=(9, 9), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    red = "#e74c3c"
    red_hist = "#e74c3c"

    # ----- 3. Main Axis: SymLog Scatter & Contour -----
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    
    log_floor = -30.0 
    vmax = float(np.max(logp))
    
    ax_main.pcolormesh(
        xx, yy, logp, 
        cmap=cmap, 
        vmin=log_floor, 
        vmax=vmax, 
        shading='auto',
        zorder=0
    )

    # SCATTER: Nur Red (Ground Truth)
    ax_main.scatter(x1_m, x2_m, s=6, alpha=0.6,
                    color="#ffd700", linewidths=0, edgecolors="none", rasterized=True, zorder=2)

    ax_main.set_xlabel(r"$x_1$", color="black")
    ax_main.set_ylabel(r"$x_2$", color="black")
    
    ax_main.set_xscale('symlog', linthresh=LIN_THRESH)
    ax_main.set_yscale('symlog', linthresh=LIN_THRESH)
    ax_main.set_xlim(-TAIL_MAX, TAIL_MAX)
    ax_main.set_ylim(-TAIL_MAX, TAIL_MAX)

    # ----- 4. Top Axis: X Marginal -----
    ax_top.set_yscale("log")
    
    # HISTOGRAM: Nur Red (gefüllt)
    # 1. Die Füllung (halbtransparent, ohne Rand)
    ax_top.hist(x2_m, bins=pts, density=True, color=red_hist, alpha=0.4, edgecolor="none")
    
    # 2. Die scharfe Kante darüberlegen (volle Deckkraft)
    ax_top.hist(x2_m, bins=pts, density=True, histtype="step", color=red_hist, linewidth=1.5)
    ax_top.tick_params(labelbottom=False)

    # ----- 5. Right Axis: Y Marginal -----
    ax_right.set_xscale("log")
    
    # HISTOGRAM: Nur Red (gefüllt)
     # RIGHT MARGINAL
    # 1. Die Füllung
    ax_right.hist(x1_m, bins=pts, density=True, orientation="horizontal",
                  color=red_hist, alpha=0.4, edgecolor="none")
    
    # 2. Die scharfe Kante darüberlegen
    ax_right.hist(x1_m, bins=pts, density=True, orientation="horizontal",
                  histtype="step", color=red_hist, linewidth=1.5)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_2)$")

    # ----- 6. Export and Log -----
    if path is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'), bbox_inches='tight')
        
    if big_eval:
        wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)

    plt.close()"""

"""
def plot_funnel_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None
    ):
    #Funnel 2D Plotting - MODIFIED: ONLY GROUND TRUTH (BRIGHT RED, STEP HISTOGRAMS)
    
    # ----- fixed axis ranges -----
    X2_MIN, X2_MAX = -999.0, 999.0   # horizontal (x-axis): x2
    X1_MIN, X1_MAX =  -20.0,    20.0   # vertical (y-axis):   x1

    n_data = generated.shape[0]
    S_data = sampler.sample(n_data,)

    # Nur echte Daten verwenden
    x1_m = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    x2_m = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    # ----- evaluate TRUE log p(x1,x2) on the fixed grid -----
    x1_lin, x2_lin, logp = compute_log_joint_grid(
        sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360
    )

    bins_x2 = np.linspace(X2_MIN, X2_MAX, 50)
    bins_x1 = np.linspace(X1_MIN, X1_MAX, 50)

    # ----- figure -----
    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05 
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    # Ein sehr helles, knalliges Rot (Vibrant Red)
    red = "#ffd700"
    red_hist = "#e74c3c"

    # main: true log-joint
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    log_floor = -20.0
    vmax = float(np.max(logp))
    ax_main.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    # SCATTER: Nur Red (Ground Truth) mit voller Deckkraft (alpha=1.0)
    ax_main.scatter(x2_m, x1_m, s=6, alpha=1.0,
                    color=red, linewidths=0, edgecolors="none", rasterized=True)

    ax_main.set_xlabel(r"$x_2$", color="white")
    ax_main.set_ylabel(r"$x_1$", color="white")
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)

    # TOP MARGINAL
    ax_top.set_yscale("log")
    
    # 1. Die Füllung (halbtransparent, ohne Rand)
    ax_top.hist(x2_m, bins=bins_x2, density=True, color=red_hist, alpha=0.4, edgecolor="none")
    
    # 2. Die scharfe Kante darüberlegen (volle Deckkraft)
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red_hist, linewidth=1.5)

    # Analytic overlay for p(x2)
    x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
    scale1 = float(getattr(sampler, 'scale1', torch.tensor(3.0)).item())
    px2 = _analytic_funnel_x2_pdf(x2_centers, scale1=scale1, gh_n=80)
    ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95, label="analytic", zorder=5)
    ax_top.tick_params(labelbottom=False)

    # RIGHT MARGINAL
    # 1. Die Füllung
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  color=red_hist, alpha=0.4, edgecolor="none")
    
    # 2. Die scharfe Kante darüberlegen
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red_hist, linewidth=1.5)
    
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    if path is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'), bbox_inches='tight')
        
    if big_eval:
      wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
      wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)

    plt.close()"""


def _sym_limits_from_arrays(a: np.ndarray, b: np.ndarray, q: float = 99.5,
                            x1_floor: float = 3.0, x1_ceil: float = 20.0,
                            x2_floor: float = 3.0, x2_ceil: float = 1000.0):
    both = np.concatenate([a, b], axis=0)
    ax1 = np.percentile(np.abs(both[:, 0]), q)
    ax2 = np.percentile(np.abs(both[:, 1]), q)
    r1 = float(np.clip(ax1, x1_floor, x1_ceil))
    r2 = float(np.clip(ax2, x2_floor, x2_ceil))
    return (-r1, r1), (-r2, r2)


def plot_generic_2d(
        generated,
        sampler,
        step,
        big_eval=False,
        path=None
    ):
    #Generic 2D plotting for targets without analytic log_prob.

    """- No density background
    - Scatter of generated samples overlaid with true samples
    - Top/right marginals as hist overlays (data red outline, model filled teal)
    - Adaptive symmetric limits from percentiles to avoid outliers dominating"""
    
    n_data = generated.shape[0]
    S_data = sampler.sample(n_data,)

    x1_d = generated[:, 0].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 0]
    x2_d = generated[:, 1].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 1]

    x1_m = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    x2_m = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    gen_np = np.stack([x1_d, x2_d], axis=-1)
    data_np = np.stack([x1_m, x2_m], axis=-1)
    (x1_min, x1_max), (x2_min, x2_max) = _sym_limits_from_arrays(gen_np, data_np)

    bins_x1 = np.linspace(x1_min, x1_max, 60)
    bins_x2 = np.linspace(x2_min, x2_max, 60)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    ax_main.scatter(x2_m, x1_m, s=4, alpha=0.25, color=red, linewidths=0, edgecolors="none", rasterized=True)
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.6, color=teal, linewidths=0, edgecolors="none", rasterized=True)
    ax_main.set_xlabel(r"$x_2$")
    ax_main.set_ylabel(r"$x_1$")
    ax_main.set_xlim(x2_min, x2_max)
    ax_main.set_ylim(x1_min, x1_max)

    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    # 1. Save your PDF for your thesis/high-quality records
    pdf_path = os.path.join(path, f'samples_epoch_{step:03d}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    # 2. Save a stable PNG for WandB to avoid the Temp folder crash
    png_path = os.path.join(path, f'samples_epoch_{step:03d}.png')
    plt.savefig(png_path) 

    if big_eval:
        # PASS THE STR PATH, NOT THE PLT OBJECT
        wandb.log({"eval/scatter_plot_big": wandb.Image(png_path)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(png_path)}, step=step)

    plt.close()

def plot_toy2d_cross(
        generated,
        sampler,
        step,
        big_eval=False,
        path=None
    ):
    """
    Custom 2D plotting for the Toy2DCross heavy-tailed distribution.
    - No density background
    - Scatter of generated samples overlaid with true samples
    - Top/right marginals as hist overlays (data red outline, model filled teal)
    - Adaptive symmetric limits from percentiles to avoid Student-t outliers dominating
    - Safely handles string or int for 'step'
    """
    
    n_data = generated.shape[0]
    # Ziehe echte Target-Samples
    S_data = sampler.sample(n_data)

    # Konvertierung zu Numpy (für x und y Dimensionen)
    x_gen = generated[:, 0].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 0]
    y_gen = generated[:, 1].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 1]

    x_true = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    y_true = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    # --- Adaptive Limits für Heavy Tails (Student-t) ---
    # Wir nehmen das 99.9. Perzentil, um die weiten Ausläufer der Student-t 
    # Verteilung zu zeigen, ohne dass ein einzelner extremer Ausreißer alles ruiniert.
    all_x = np.concatenate([x_gen, x_true])
    all_y = np.concatenate([y_gen, y_true])
    
    max_val = np.percentile(np.abs(np.concatenate([all_x, all_y])), 99.99)
    
    # Deutlich weiter herausgezoomt: Mindestens [-15, 15]
    max_val = max(25.0, max_val) 
    
    lim = (-max_val, max_val)
    bins = np.linspace(lim[0], lim[1], 80)

    # --- Setup GridSpec ---
    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    # --- Main Scatter ---
    # Target in Rot (unten), Generiert in Teal (darüber)
    ax_main.scatter(x_true, y_true, s=4, alpha=0.25, color=red, linewidths=0, edgecolors="none", rasterized=True, label="Target")
    ax_main.scatter(x_gen, y_gen, s=6, alpha=0.4, color=teal, linewidths=0, edgecolors="none", rasterized=True, label="Generated")
    ax_main.set_xlabel(r"$x_1$")
    ax_main.set_ylabel(r"$x_2$")
    ax_main.set_xlim(lim)
    ax_main.set_ylim(lim)
    ax_main.legend(loc="upper left")

    # --- Top Hist (Marginal X) ---
    ax_top.hist(x_true, bins=bins, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x_gen, bins=bins, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.tick_params(labelbottom=False)

    # --- Right Hist (Marginal Y) ---
    ax_right.hist(y_gen, bins=bins, density=True, orientation="horizontal", color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(y_true, bins=bins, density=True, orientation="horizontal", histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_2)$")

    # --- Safe Step Formatting (Behebt den ValueError bei f"{step}_target") ---
    if isinstance(step, int):
        step_str = f"{step:03d}"
    else:
        step_str = str(step)

    # 1. Speichere PDF
    if path is not None:
        os.makedirs(path, exist_ok=True)
        pdf_path = os.path.join(path, f'samples_epoch_{step_str}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')

        # 2. Speichere PNG & Logge zu WandB
        png_path = os.path.join(path, f'samples_epoch_{step_str}.png')
        plt.savefig(png_path, bbox_inches='tight') 

        log_key = "eval/scatter_plot_big" if big_eval else "eval/scatter_plot"
        wandb.log({f"{log_key}_toy2d_cross": wandb.Image(png_path)}, step=step if isinstance(step, int) else None)

    plt.close()

def plot_thin_angles(
        generated,
        sampler,
        step,
        big_eval=False,
        path=None
    ):
    #thin angles 2D plotting for targets without analytic log_prob.

    """- No density background
    - Scatter of generated samples overlaid with true samples
    - Top/right marginals as hist overlays (data red outline, model filled teal)
    - Adaptive symmetric limits from percentiles to avoid outliers dominating"""
    
    n_data = generated.shape[0]
    S_data = sampler.sample(n_data,)

    x1_d = generated[:, 0].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 0]
    x2_d = generated[:, 1].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 1]

    x1_m = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    x2_m = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    gen_np = np.stack([x1_d, x2_d], axis=-1)
    data_np = np.stack([x1_m, x2_m], axis=-1)
    (x1_min, x1_max), (x2_min, x2_max) = (-8,8), (-8,8)

    bins_x1 = np.linspace(x1_min, x1_max, 60)
    bins_x2 = np.linspace(x2_min, x2_max, 60)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    ax_main.scatter(x2_m, x1_m, s=4, alpha=0.25, color=red, linewidths=0, edgecolors="none", rasterized=True)
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.6, color=teal, linewidths=0, edgecolors="none", rasterized=True)
    ax_main.set_xlabel(r"$x_2$")
    ax_main.set_ylabel(r"$x_1$")
    ax_main.set_xlim(x2_min, x2_max)
    ax_main.set_ylim(x1_min, x1_max)

    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    # 1. Save your PDF for your thesis/high-quality records
    pdf_path = os.path.join(path, f'samples_epoch_{step:03d}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    # 2. Save a stable PNG for WandB to avoid the Temp folder crash
    png_path = os.path.join(path, f'samples_epoch_{step:03d}.png')
    plt.savefig(png_path) 

    if big_eval:
        # PASS THE STR PATH, NOT THE PLT OBJECT
        wandb.log({"eval/scatter_plot_big": wandb.Image(png_path)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(png_path)}, step=step)

    plt.close()

"""def plot_generic_2d(
        generated,
        sampler,
        step,
        big_eval=False,
        path=None
    ):
    #Generic 2D plotting - MODIFIED: ONLY GROUND TRUTH (RED)
    n_data = generated.shape[0]
    S_data = sampler.sample(n_data,)

    # x1_d und x2_d berechnen wir zwar noch, plotten sie aber nicht mehr.
    x1_d = generated[:, 0].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 0]
    x2_d = generated[:, 1].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 1]

    x1_m = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    x2_m = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    # gen_np = np.stack([x1_d, x2_d], axis=-1)  <-- Brauchen wir nicht mehr zwingend
    data_np = np.stack([x1_m, x2_m], axis=-1)
    
    # ANPASSUNG: Limits nur noch basierend auf data_np berechnen!
    (x1_min, x1_max), (x2_min, x2_max) = _sym_limits_from_arrays(data_np, data_np)

    bins_x1 = np.linspace(x1_min, x1_max, 60)
    bins_x2 = np.linspace(x2_min, x2_max, 60)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    # MAIN PLOT: Nur Red
    ax_main.scatter(x2_m, x1_m, s=4, alpha=0.25, color=red, linewidths=0, edgecolors="none", rasterized=True)
    # ax_main.scatter(x2_d, x1_d, s=6, alpha=0.6, color=teal, linewidths=0, edgecolors="none", rasterized=True) <-- ENTFERNT

    ax_main.set_xlabel(r"$x_2$")
    ax_main.set_ylabel(r"$x_1$")
    ax_main.set_xlim(x2_min, x2_max)
    ax_main.set_ylim(x1_min, x1_max)

    # TOP MARGINAL: Nur Red
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    # ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal) <-- ENTFERNT
    ax_top.tick_params(labelbottom=False)

    # RIGHT MARGINAL: Nur Red
    # ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal", color=teal, alpha=0.35, edgecolor=teal) <-- ENTFERNT
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal", histtype="step", color=red, linewidth=2.0)
    
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'), bbox_inches='tight')
    if big_eval:
        wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)
    plt.close()"""


@torch.no_grad()
def plot_latent_colored_by_target_norm(
    latent: torch.Tensor,           # eps at t=1, shape (N,2)
    targets: torch.Tensor,          # generated x at t=0, shape (N,2)
    step: int,
    path: str,
    big_eval: bool = False,
    title: str = "Latent colored by ||x||",
):
    """
    Two-panel figure:
    - Left: raw Gaussian samples in latent space.
    - Right: same latent points colored by the norm of their reached target.

    Logged to W&B under eval keys (big vs light eval differentiated by suffix).
    """
    os.makedirs(path or ".", exist_ok=True)

    L = latent.detach().cpu().numpy() if isinstance(latent, torch.Tensor) else latent
    X = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    import numpy as np
    norms = np.linalg.norm(X, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150, constrained_layout=True)

    # Left: plain latent Gaussian
    ax0 = axes[0]
    latent_color = COL_DENSITY if big_eval else "#808080"
    marker_size = 3 if big_eval else 4
    latent_alpha = 0.22 if big_eval else 0.4
    ax0.scatter(L[:, 0], L[:, 1], s=marker_size, alpha=latent_alpha, color=latent_color, linewidths=0)
    ax0.set_title("Latent Gaussian (t=1)")
    ax0.set_aspect('equal', 'box')
    if big_eval:
        ax0.grid(False)
    else:
        ax0.grid(True, alpha=0.2)

    # Right: latent colored by ||x||
    ax1 = axes[1]
    cmap = "afmhot" if big_eval else "viridis"
    color_alpha = 0.6 if big_eval else 0.7
    h = ax1.scatter(L[:, 0], L[:, 1], s=5, c=norms, cmap=cmap, alpha=color_alpha, linewidths=0)
    default_title = "Latent colored by ||x||"
    effective_title = "Latent Colored by ||x||" if title.strip().lower() == default_title.lower() else title
    ax1.set_title(effective_title)
    ax1.set_aspect('equal', 'box')
    if big_eval:
        ax1.grid(False)
    else:
        ax1.grid(True, alpha=0.2)
    cbar = fig.colorbar(h, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("||x||")

    out_name = f'latent_color_epoch_{int(step):03d}.png'
    # 1. Save the exact path to a variable
    saved_image_path = os.path.join(path or ".", out_name)

    # 2. Save the figure to that path (your code already did this perfectly)
    fig.savefig(saved_image_path, bbox_inches='tight', dpi=150)

    wb_key = "eval/latent_color_big" if big_eval else "eval/latent_color"

    # 3. Pass the PATH string to wandb.Image(), NOT the fig object!
    wandb.log({wb_key: wandb.Image(saved_image_path)}, step=step)

    plt.close(fig)


def plot_vector_field(model, device, t=0.0, path=None, step = None):
    """
    Plottet das Vektorfeld des Modells zu einem bestimmten Zeitpunkt t.
    """
    model.eval()
    
    # Gitter definieren (angepasst an deinen Plot-Bereich)
    # Wir schauen uns bewusst den "Problembereich" an (z.B. X von -750 bis 750, Y von -20 bis 20)
    # ACHTUNG: Wir erstellen das Gitter im ECHTEN Raum und skalieren es dann für das Modell!
    y = np.linspace(-500, 500, 30) 
    x = np.linspace(-20, 20, 30)
    X, Y = np.meshgrid(x, y)
    
    # Flatten für das Modell
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    # ---------------------------------------------------------
    # WICHTIG: Skalierung anwenden, wie im Training!
    # ---------------------------------------------------------
    # Wir tun so, als wären diese Gitterpunkte unsere "Noise/Data"
    norms = torch.linalg.norm(grid_tensor, dim=1, keepdim=True)
    scale = torch.log1p(norms) / (norms + 1e-8)
    scaled_grid = grid_tensor * scale
    
    # Zeit-Tensor erstellen
    t_tensor = torch.ones(len(scaled_grid), 1).to(device) * t
    
    with torch.no_grad():
        # Modell vorhersage im SKALIERTEN Raum
        v_pred_scaled = model(t_tensor, scaled_grid)
        
        # Vektoren zurück in den echten Raum rechnen? 
        # Eigentlich reicht es, die Richtung im skalierten Raum zu sehen, 
        # aber für die Interpretation ist die Rückskalierung besser.
        # (Hier vereinfacht: Wir plotten die rohen Vektoren des Modells auf dem Gitter)
        # Wenn v_pred_scaled fast 0 ist, "weiß" das Modell nichts.
        
    U = v_pred_scaled[:, 0].cpu().numpy().reshape(X.shape)
    V = v_pred_scaled[:, 1].cpu().numpy().reshape(Y.shape)
    
    # Plotten
    magnitude = np.sqrt(U**2 + V**2) + 1e-8
    U_dir = U / magnitude
    V_dir = V / magnitude
    
    # Plotten
    plt.figure(figsize=(10, 8))
    plt.title(f"Das Gehirn des Modells bei t={t:.2f} (Windrichtung)")
    
    # NEU: aspect='auto' zwingt das Bild in eine normale Form
    # Passe die extent-Werte an deine echten Achsen an (hier für X=-20..20, Y=-800..800)
    plt.imshow(np.log1p(np.sqrt(X**2 + Y**2)), 
               extent=[-20, 20, -500, 500], 
               origin='lower', cmap='magma', alpha=0.3, aspect='auto')
    
    # NEU: angles='xy' und normierte Vektoren für saubere Pfeile
    # color='black' (oder 'white', je nachdem was besser lesbar ist)
    plt.quiver(X, Y, U_dir, V_dir, color='black', alpha=0.7, 
               angles='xy', pivot='middle', scale=25)
    
    plt.xlabel("X (Echter Raum)")
    plt.ylabel("Y (Echter Raum)")
    
    # Hier auch anpassen!
    plt.xlim(-20, 20)
    plt.ylim(-500, 500)
    plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}_t_{t:.2f}.pdf'))
    plt.close()

# Aufruf: Schau dir den Start (t=0) und kurz vor Ende (t=0.9) an
# plot_vector_field(model, device, t=0.0) 
# plot_vector_field(model, device, t=0.9)

def inspect_single_trajectory(model, start_noise, t_vals, path=None, step = None):
    """
    Verfolgt einen Punkt und printet seine Werte, um zu sehen, wo er abbiegt.
    start_noise: Tensor shape (1, 2)
    """
    # 1. Skalieren
    r_start_real = start_noise.norm(dim=1, keepdim=True) + 1e-8
    r_start_log = torch.log1p(r_start_real)
    scale = torch.log1p(start_noise.norm(dim=1, keepdim=True)) / (start_noise.norm(dim=1, keepdim=True) + 1e-8)
    current_pos = start_noise * scale
    
    trajectory = [current_pos.cpu().numpy()]
    
    print("--- Starte Inspektion ---")
    print(f"Start (Real): {start_noise.cpu().numpy()}")
    print(f"Start (Scaled): {current_pos.cpu().numpy()}")
    
    dt = t_vals[1] - t_vals[0]
    
    for t in t_vals[:-1]:
        t_tensor = torch.ones(1, 1).to(start_noise.device) * t
        with torch.no_grad():
            vel = model(t_tensor, current_pos)
        
        # Euler Step (simpel)
        current_pos = current_pos + vel * dt
        current_radius = current_pos.norm(dim=1, keepdim=True) + 1e-8
        current_pos = (current_pos / current_radius) * r_start_log
        
        trajectory.append(current_pos.cpu().numpy())
        
        # Check: Ist die Velocity verdächtig klein?
        vel_norm = vel.norm().item()
        if vel_norm < 1e-4:
            print(f"WARNUNG bei t={t:.2f}: Velocity fast 0 ({vel_norm:.5f})! Punkt bleibt stecken.")
            
    trajectory = np.array(trajectory).squeeze()
    trajectory = trajectory / scale.cpu().numpy()  # Rückskalieren für Interpretation
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(trajectory[:,0], trajectory[:,1], 'r.-', label='Pfad im skalierten Raum')
    plt.scatter(trajectory[0,0], trajectory[0,1], c='green', s=100, label='Start')
    plt.scatter(trajectory[-1,0], trajectory[-1,1], c='purple', s=100, label='Ende')
    plt.legend()
    plt.title("Wanderung eines verlorenen Punktes")
    plt.grid(True)
    plt.savefig(os.path.join(path, f'trajectory_epoch_{step:03d}_point_{start_noise.cpu().numpy()[0][0]:.1f}_{start_noise.cpu().numpy()[0][1]:.1f}.pdf'))
    plt.close()

def plot_dead_zone_velocity(model, x_value, device, path, step):
    # Erzeuge Punkte auf einer vertikalen Linie bei X=300
    y_vals = torch.linspace(-15, 15, 100, device=device)
    x_vals = torch.full_like(y_vals, x_value)
    test_points = torch.stack([x_vals, y_vals], dim=1)
    
    # Skalieren für das Modell (Log-Raum)
    r_real = test_points.norm(dim=1, keepdim=True)
    r_log = torch.log1p(r_real)
    test_points_scaled = test_points / r_real * r_log
    
    # Zeit t=1.0 (Start der Generierung)
    t_tensor = torch.ones(100, 1, device=device)
    
    with torch.no_grad():
        vel = model(t_tensor, test_points_scaled)
    
    # Da wir uns für die Drehung interessieren, berechnen wir die Winkelgeschwindigkeit (Omega)
    # v_tangential_magnitude = vel * Tangentenvektor
    tangent = torch.stack([-test_points_scaled[:,1], test_points_scaled[:,0]], dim=1)
    tangent = tangent / tangent.norm(dim=1, keepdim=True)
    omega = (vel * tangent).sum(dim=1).cpu().numpy()
    
    plt.figure(figsize=(8, 4))
    plt.plot(y_vals.cpu().numpy(), omega, 'b-', linewidth=2)
    plt.axvline(0, color='r', linestyle='--', label='Der Cut (Y=0)')
    plt.axhline(0, color='k', linestyle='-')
    plt.xlabel(f'Y-Startposition (bei X={x_value})')
    plt.ylabel('Initiale Winkelgeschwindigkeit \omega')
    plt.title('Die "Dead Zone" des Vektorfeldes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f'dead_zone_epoch_{step:03d}_x_{x_value}.pdf'))
    plt.close()
