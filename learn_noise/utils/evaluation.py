import torch
import wandb
from tqdm import tqdm
from torchdiffeq import odeint
from learn_noise.utils.spherical_ode import slerp_tvdrk3, spherical_ode_solver
from learn_noise.utils.mmd import memory_efficient_mmd
from typing import Dict, Tuple
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import numpy as np

import learn_noise.utils.plotting as plot

_MAX_SEED = 2 ** 31 - 1


def _get_seed(base_seed: int, offset: int) -> int:
    seed = (base_seed + offset) % _MAX_SEED
    if seed <= 0:
        seed += 1
    return seed


def _get_target_cache(args) -> Dict[Tuple[str, int], torch.Tensor]:
    if not hasattr(args, "_eval_fixed_targets") or args._eval_fixed_targets is None:
        setattr(args, "_eval_fixed_targets", {})
    return args._eval_fixed_targets


def _get_uniform_cache(args) -> Dict[Tuple[int, int], torch.Tensor]:
    if not hasattr(args, "_eval_fixed_u") or args._eval_fixed_u is None:
        setattr(args, "_eval_fixed_u", {})
    return args._eval_fixed_u


def _to_raw_if_needed(sampler, tensor: torch.Tensor) -> torch.Tensor:
    return sampler.to_raw(tensor) if hasattr(sampler, "to_raw") else tensor


def _base_sampler(sampler):
    return sampler.base if hasattr(sampler, "base") else sampler


def _fixed_ground_truth(args, sampler, raw: bool, total: int, device: torch.device) -> torch.Tensor:
    cache = _get_target_cache(args)
    target_dataset = args.target_dataset if hasattr(args, "target_dataset") else "unknown"
    key = (target_dataset, total)
    if key not in cache:
        base_seed = int(args.seed) if hasattr(args, "seed") else 0
        seed = _get_seed(base_seed, 1009 + 131 * total)
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            samples = sampler.sample(total, device=device, dtype=torch.float32)
            if raw:
                samples = _to_raw_if_needed(sampler, samples).cpu()
        cache[key] = samples
    return cache[key]


def _fixed_uniform(args, total: int, dim: int, offset: int = 0) -> torch.Tensor:
    cache = _get_uniform_cache(args)
    key = (total, dim)
    if key not in cache:
        base_seed = int(args.seed) if hasattr(args, "seed") else 0
        seed = _get_seed(base_seed, 2027 + 137 * total + offset)
        gen = torch.Generator()
        gen.manual_seed(seed)
        cache[key] = torch.rand((total, dim), generator=gen, dtype=torch.float32)
    return cache[key]

def log_distributions_to_wandb(x_gen_scaled, gt_scaled, target_name, step):
    """
    Creates a side-by-side 2D scatter plot of ground truth vs generated data 
    and logs it directly to Weights & Biases.
    """
    # 1. Safely convert PyTorch tensors to NumPy arrays
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    gen_np = to_numpy(x_gen_scaled)
    gt_np = to_numpy(gt_scaled)

    # 2. Create figure with 2 side-by-side subplots
    # sharex and sharey ensure they stay on the exact same scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Optional but highly recommended: Find global min/max to set fixed limits
    x_min = min(gt_np[:, 0].min(), gen_np[:, 0].min())
    x_max = max(gt_np[:, 0].max(), gen_np[:, 0].max())
    y_min = min(gt_np[:, 1].min(), gen_np[:, 1].min())
    y_max = max(gt_np[:, 1].max(), gen_np[:, 1].max())
    
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    
    ax1.set_xlim(x_min - margin_x, x_max + margin_x)
    ax1.set_ylim(y_min - margin_y, y_max + margin_y)
    
    # 3. Plot Ground Truth (Left)
    ax1.scatter(gt_np[:, 0], gt_np[:, 1], alpha=0.5, color='blue', s=10)
    ax1.set_title("Ground Truth")
    ax1.grid(True, alpha=0.3)

    # 4. Plot Generated (Right)
    ax2.scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, color='red', s=10)
    ax2.set_title("Generated")
    ax2.grid(True, alpha=0.3)

    # Main title for the entire figure
    fig.suptitle(f"2D Distribution: {target_name} (Step {step})", fontsize=14)
    plt.tight_layout()

    # 5. Log the combined figure to wandb
    wandb.log({f"eval/plot_{target_name}": wandb.Image(fig)}, step=step)

    # 6. CRITICAL: Close the figure to prevent RAM leaks
    plt.close(fig)


@torch.no_grad()
def heavy_eval_batched(
    args,
    x_0, 
    ode_func, 
    sampler,
    step, 
    big_eval=False,
    device='cpu', 
    noise = None,
    quantile=None,
):
    """
    Massive eval to probe tails with VRAM-safe batching.
    - Generates eps at τ=1 in batches (Student-t base)
    - Integrates ODE to t=0, collects running NLL mean
    - Keeps a capped subset for plotting (both latent eps and generated x)
    - Logs GeomLoss Sinkhorn/MMD metrics on cached subsets for non-funnel targets
    """
    dim = args.dim
    output_dir = args.runs_dir

    device = torch.device(device)

    if big_eval:
        total    = int(args.big_eval_samples)
    else: 
        total    = int(args.eval_sample)

    if total <= 0:
        return

    batch_size    = int(args.eval_batch)

    keep  = total#int(args.eval_plot_samples)
    assert batch_size > 0, "big_eval_batch must be > 0"

    t_vals = torch.linspace(1, 0.0, args.num_steps_eval, device=device)

    nll_sum = 0.0
    seen = 0

    kept_x = []
    kept_eps = []
    kept_count = 0

    u_unit_cache = None
    if quantile is not None and total > 0:
        u_unit_cache = _fixed_uniform(args, total, 1) #dim

    target_name = (args.target_dataset if hasattr(args, "target_dataset") else "funnel").lower()
    raw_sampler = _base_sampler(sampler)

    # progress loop
    num_loops = (total + batch_size - 1) // batch_size
    for loop_idx in range(num_loops):
        current_batch_size = min(batch_size, total - seen)
        if current_batch_size <= 0:
            break
        # Initial noise at τ=1: prefer quantile if provided for consistency
        if quantile is not None:
            u_eps = float(args.q_u_eps) if hasattr(args, "q_u_eps") else 5e-5
            u_slice = u_unit_cache[seen: seen + current_batch_size].to(device)
            Uv = u_eps + (1 - 2 * u_eps) * u_slice
            ones_t = torch.ones(current_batch_size, 1, device=device)
            with torch.no_grad():
                eps = quantile(Uv, ones_t, asRad = True)
            uni = torch.randn(size = (eps.shape[0], args.dim), device = device)
            uni = torch.nn.functional.normalize(uni)
            eps = eps * uni
        elif noise is not None:
            if args.baseline_flow == "target_norm":
                x0 = sampler.sample(current_batch_size, device=device, dtype=torch.float32)
                eps = noise((current_batch_size, dim), x0).to(device) #dim
            else:
                eps = noise((current_batch_size, dim)).to(device) #dim
                print(max(eps.norm(dim=1)).item())
        else:
            eps = torch.randn(current_batch_size, dim, device=device)

        x_T = eps 

        if args.slerp:
            scale = torch.log1p(x_T.norm(dim=1, keepdim=True)) / (x_T.norm(dim=1, keepdim=True) + 1e-8)
            x_T_scaled = x_T * scale  # Logarithmische Skalierung der Start
            trajectory = spherical_ode_solver(ode_func, x_T_scaled, t_vals)
            trajectory = trajectory / (scale + 1e-8)  # Rückskalierung der Trajektorie auf lineare Geschwindigkeit
        else:
            trajectory = odeint(ode_func, x_T, t_vals, method="dopri5")

        x_gen = trajectory[-1]   # (cur_bs, dim)
        if not big_eval:
            max_radius_error = (x_T.norm(dim=1) - x_gen.norm(dim=1)).abs().max().item()
            print(f"[Sanity Check] Maximum radial drift across all points: {max_radius_error:.8f}")

        # Accumulate NLL sum to compute global mean at the end
        #nll_sum += (-sampler.log_prob(x_gen)).sum().item()
        seen += current_batch_size

        # Keep a proportionate random subset from this batch for plotting
        per_batch_keep = max(1, int(round(keep * (current_batch_size / total)))) if keep > 0 else 0
        if per_batch_keep > 0:
            #idx = torch.randperm(current_batch_size, device=device)[:per_batch_keep]
            kept_x.append(x_gen.detach().cpu())
            kept_eps.append(eps.detach().cpu())
            kept_count += per_batch_keep

    x_gen = torch.stack(kept_x, dim=0).reshape(-1, dim) #dim
    x_gen_raw = _to_raw_if_needed(sampler, x_gen)
    eps_kept = torch.stack(kept_eps, dim=0).reshape(-1, dim) if kept_eps else None #dim
    
    '''# Plot (downsample to exactly 'keep' if we slightly overshot)
    if keep > 0 and kept_x:
        X = torch.cat(kept_x, dim=0)
        E = torch.cat(kept_eps, dim=0)
        if X.shape[0] > keep:
            perm = torch.randperm(X.shape[0])[:keep]
            X = X[perm]
            E = E[perm]'''

    # sinkhorn evalutation
    import scipy.stats
    import numpy as np

    # =====================================================================
    # PREPARATION: Separate Pools for Global (MMD/OT) vs. Tail (KS) Metrics
    # =====================================================================
    # 1. Global Metrics Pool (Max 50k random samples to preserve the bulk)
    MAX_SAMPLES_GLOBAL = min(50000, x_gen.shape[0])
    curr_x_gen_global = x_gen[:MAX_SAMPLES_GLOBAL].to(device)
    
    # FIX: Nutze 'sampler' statt 'raw_sampler', damit die GT auch Z-Scores sind!
    curr_gt_global = _fixed_ground_truth(args, sampler, False, curr_x_gen_global.shape[0], device=device).to(device)

    # 2. Tail Metrics Pool (Use the ENTIRE generated batch for maximum tail resolution)
    N_total = x_gen.shape[0]
    print(f"Total generated samples for tail metrics: {N_total}")
    
    # FIX: Auch hier 'sampler' nutzen für Z-Scores!
    full_gt_samples = _fixed_ground_truth(args, sampler, False, N_total, device=device).to(device)

    print(f"Hat x_gen NaNs oder Infs? {not torch.isfinite(x_gen).all().item()}")

    # =====================================================================
    # QUICK SANITY CHECK: Z-Scores vs. Raw Data & NaNs
    # =====================================================================
    print("\n" + "="*50)
    print("SANITY CHECK: Z-SCORES VS RAW DATA")
    
    # 1. Check auf NaNs und Infs
    gen_is_finite = torch.isfinite(x_gen).all().item()
    gt_is_finite = torch.isfinite(full_gt_samples).all().item()
    print(f"x_gen ist finite (keine NaNs/Infs)?       {gen_is_finite}")
    print(f"full_gt_samples ist finite?               {gt_is_finite}")

    # 2. Globale Statistiken (Mean, Std, Min, Max)
    print("-" * 50)
    print(f"x_gen           | Mean: {x_gen.mean().item():>8.3f} | Std: {x_gen.std().item():>8.3f} | Min: {x_gen.min().item():>10.3f} | Max: {x_gen.max().item():>10.3f}")
    print(f"full_gt_samples | Mean: {full_gt_samples.mean().item():>8.3f} | Std: {full_gt_samples.std().item():>8.3f} | Min: {full_gt_samples.min().item():>10.3f} | Max: {full_gt_samples.max().item():>10.3f}")
    print("="*50 + "\n")
    

    # =====================================================================
    # 1. SINKHORN DISTANCE (Skip for funnel if numerical instability is too high, 
    # but strictly evaluating on z-scores usually makes it safe!)
    # =====================================================================
    if target_name not in {"radialpareto", "funnel"}: # Re-enabled for funnel assuming z-scores!
        z = torch.cat((curr_x_gen_global, curr_gt_global), dim=0)
        offset = z.mean(dim=0)
        scale = 10 * (z - offset).abs().mean().detach() + 1e-6
        
        x_gen_scaled = (curr_x_gen_global - offset) / scale
        gt_scaled = (curr_gt_global - offset) / scale

        log_distributions_to_wandb(curr_x_gen_global, curr_gt_global, target_name, step)
        
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05/scale, scaling=0.9, backend = "tensorized")
        sinkhorn_dist = loss_fn(x_gen_scaled, gt_scaled)       
        wandb.log({f"eval/sinkhorn_{target_name}": sinkhorn_dist.item()}, step=step)
    """if target_name not in {"radialpareto", "funnel"}: # Re-enabled for funnel assuming z-scores!
        z = torch.cat((curr_x_gen_global, curr_gt_global), dim=0)
        offset = z.mean(dim=0)
        scale = 10 * (z - offset).abs().mean().detach() + 1e-6
        
        x_gen_scaled = (curr_x_gen_global - offset) / scale
        gt_scaled = (curr_gt_global - offset) / scale
        
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05/scale, scaling=0.9)
        # Move to CPU for the numpy-dependent geomloss evaluation, then back to device
        # 1. Force geomloss internal parameters to be standard floats so NumPy doesn't panic
        # 1. Force geomloss internal parameters to be standard floats
        if hasattr(loss_fn, 'blur'):
            loss_fn.blur = float(loss_fn.blur)
        if hasattr(loss_fn, 'scaling'):
            loss_fn.scaling = float(loss_fn.scaling)

        # 2. CAP THE SAMPLES TO 10,000 TO PREVENT 10GB RAM EXPLOSIONS
        max_eval = 100000
        x_eval_safe = x_gen_scaled[:max_eval].cpu()
        gt_eval_safe = gt_scaled[:max_eval].cpu()

        # 3. Run the evaluation safely
        sinkhorn_dist = loss_fn(x_eval_safe, gt_eval_safe).to(x_gen_scaled.device)  
        wandb.log({f"eval/sinkhorn_{target_name}": sinkhorn_dist.item()}, step=step)"""

    # =====================================================================
    # 2. MAXIMUM MEAN DISCREPANCY (MMD)
    # =====================================================================
    # Uses the memory-efficient 50k global subset
    mmd_val = memory_efficient_mmd(curr_x_gen_global, curr_gt_global, chunk_size=2000, target_name=target_name, step=step)
    wandb.log({f"eval/mmd_{target_name}": mmd_val}, step=step)

    # =====================================================================
    # 3. HIGH-RESOLUTION ROBUST TAIL KS TEST
    # =====================================================================
    # Move the FULL tensors to CPU for SciPy
    x_gen_np = x_gen.detach().cpu().numpy()
    gt_np = full_gt_samples.detach().cpu().numpy()

    # With the full batch (e.g., 100k - 500k), 1% gives us a beautiful, high-res tail
    tail_fraction = 0.01 

    # --- VARIANT A: Robust Flat Tail KS (Highly relevant for Funnel!) ---
    # Funnel is highly anisotropic (normal in v, heavy in x). Flattening mixes them, 
    # but it still rigorously tests if the network captures the extreme marginals.
    x_gen_flat = x_gen_np.flatten()
    gt_flat = gt_np.flatten()

    k_flat = int(tail_fraction * len(gt_flat))

    if k_flat > 0:
        gen_flat_sorted = np.sort(x_gen_flat)
        gt_flat_sorted = np.sort(gt_flat)

        gen_right = gen_flat_sorted[-k_flat:]
        gt_right = gt_flat_sorted[-k_flat:]
        ks_right = scipy.stats.ks_2samp(gen_right, gt_right).statistic

        gen_left = gen_flat_sorted[:k_flat]
        gt_left = gt_flat_sorted[:k_flat]
        ks_left = scipy.stats.ks_2samp(gen_left, gt_left).statistic

        ks_tail_avg = (ks_right + ks_left) / 2.0
    else:
        ks_tail_avg = 1.0

    wandb.log({f"eval/tail_ks_flat_{target_name}": ks_tail_avg}, step=step)

    # --- VARIANT B: Robust Radial Tail KS ---
    # Note for Funnel: Since the funnel is not isotropic, standard radial norms 
    # are dominated by the x-dimensions. This is still a valid test for extreme 
    # spatial escapes, though Flat KS is generally preferred here.
    r_gen = np.linalg.norm(x_gen_np, axis=1)
    r_gt = np.linalg.norm(gt_np, axis=1)

    k_radial = int(tail_fraction * len(r_gt))

    if k_radial > 0:
        r_gen_sorted = np.sort(r_gen)
        r_gt_sorted = np.sort(r_gt)

        r_gen_top_k = r_gen_sorted[-k_radial:]
        r_gt_top_k = r_gt_sorted[-k_radial:]

        ks_tail_radial = scipy.stats.ks_2samp(r_gen_top_k, r_gt_top_k).statistic
    else:
        ks_tail_radial = 1.0

    wandb.log({f"eval/tail_ks_radial_{target_name}": ks_tail_radial}, step=step)
    if target_name in {"funnel", "nealfunnel"}:
        plot.plot_funnel_tail_ccdf(args, raw_sampler, x_gen_raw.norm(dim=1), step = step, filename="funnel_ccdf", key="tail_ccdf")
        plot.plot_funnel_2d(x_gen_raw, raw_sampler, step, big_eval, output_dir)
    elif target_name == "radialpareto":
        plot.plot_pareto_2d(x_gen_raw, sampler, step, big_eval, output_dir)
    elif target_name == "thinangles":
        plot.plot_thin_angles(x_gen_raw, sampler, step, big_eval, output_dir)
    else:
        plot.plot_generic_2d(x_gen, sampler, step, big_eval, output_dir)
    #print(funnel_eval.evaluate_x2_marginal_metrics(x_gen))
    
    # New: latent colored by norm of reached target x
    if eps_kept is not None:
        plot.plot_latent_colored_by_target_norm(eps_kept, x_gen_raw, step, output_dir, big_eval=big_eval)
