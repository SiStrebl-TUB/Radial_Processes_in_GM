
from __future__ import annotations

import ast
import math, re
import os
import time
import ot
from typing import Callable, Optional
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import scipy

import learn_noise.utils.sampler as smpl
from learn_noise.networks.model_wrapper import TorchWrapper, ODEWrapper, AngularVelocityWrapper, AngularInferenceWrapper
from learn_noise.training.common import (
    seed_all,
    make_fixed_sampler,
    minibatch_ot_pairing,
    count_parameters,
    write_model_size_summary,
    spherical_ot_pairing,
    oversample_heavy_tails
)
from learn_noise.training.logging import (
    log_baseline_evaluation,
    log_baseline_image_metrics,
    log_real_rgb_histogram_once,
)
from learn_noise.utils import plotting
from learn_noise.utils.image_eval import reshape_flat_samples
from learn_noise.utils.velocity_kac import compute_velocity_kac
from learn_noise.utils.velocity_mmd import compute_velocity_mmd
from learn_noise.training.sample_utils import generate_baseline_samples


class MSGMSpoofWrapper(torch.nn.Module):
    """
    Tricks your Flow Matching evaluation scripts into working with the MSGM SDE.
    """
    def __init__(self, msgm_sde_model, device):
        super().__init__()
        self.model = msgm_sde_model
        self.device = device
        
        # We need this so your code doesn't crash if it tries to check the device
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @property
    def _base_model(self):
        # EMA models hide the actual network inside '.module'
        return self.model.module if hasattr(self.model, "module") else self.model

    def forward(self, t, x):
        # SAFEST FIX: Return zeros for the deterministic ODE visualizer.
        # MSGM generates samples via a stochastic SDE (Euler-Maruyama), not an ODE.
        # Returning zeros prevents the `odeint` crash and lets the script move
        # safely to the actual generation metrics (which use the .sample() method below).
        return torch.zeros_like(x)

    @torch.no_grad()
    def sample(self, num_samples, *args, **kwargs):
        # Intercept generation calls and route them to MSGM's reverse sampler
        from learn_noise.msgm_lib.sde_scheme import euler_maruyama_sampler
        
        # 1. Generate initial noise
        # Note: If MSGM is pulling everything to a specific radius in the latent space,
        # you might need to adjust the '* 5.0' to match their expected prior!
        x_noise = torch.randn(num_samples, self._base_model.base_sde.dim, device=self.device)
        # x_noise = torch.nn.functional.normalize(x_noise) * 5.0 
        
        # 2. Run their reverse diffusion
        # FIX: self._base_model IS the reverse SDE. Pass it directly!
        xs = euler_maruyama_sampler(self._base_model, x_noise, num_steps=128, keep_all_samples=False, normCorrection=True)
        
        return xs[-1] # Return the final denoised samples

def plot_neals_funnel(x_samples, y_samples, x_true=None, y_true=None, save_path="funnel_plot.png"):
    """
    Erstellt den Funnel-Plot im gewünschten Dark-Style.
    
    Parameter:
    - x_samples, y_samples: Die generierten Punkte (hellblau).
    - x_true, y_true: (Optional) Die echten Trainingsdaten für den lila Hintergrund.
    - save_path: Dateiname für das gespeicherte Bild.
    """
    # Plot aufsetzen (quadratisch sieht beim Funnel oft gut aus)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Hintergrund des Graphen auf Schwarz setzen
    ax.set_facecolor('black')
    
    # 1. Den lila Dichte-Hintergrund plotten (falls echte Daten übergeben wurden)
    if x_true is not None and y_true is not None:
        # 'magma' oder 'plasma' erzeugen diesen schönen lila/pinken Glow auf Schwarz
        # Vmax etwas herabsetzen, damit auch dünnere Bereiche gut leuchten
        ax.hist2d(x_true, y_true, bins=150, cmap='magma', density=True, alpha=0.7, vmax=0.005)
        
    # 2. Die generierten / zu untersuchenden Punkte als hellblauen Scatter plotten
    # s=5 (Größe), alpha=0.6 (Transparenz für den Überlagerungseffekt)
    ax.scatter(x_samples, y_samples, s=4, c='#88c0d0', alpha=0.6, edgecolors='none')
    
    # Achsen-Limits festlegen (ähnlich wie in deinen Bildern)
    ax.set_xlim([-900, 900])
    ax.set_ylim([-20, 20])
    
    # Styling der Achsen und Ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Einen sauberen schwarzen Rahmen um den Plot ziehen
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
        
    # Plot eng zuschneiden und in hoher Qualität speichern
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_slerp_warped(t, omega, p=3.0):
    """
    Computes warped SLERP schedule and its derivatives.
    """
    # Calculate the warped time and its derivative
    t_warped = t ** p
    t_deriv = p * (t ** (p - 1.0))
    
    # Precompute the denominator for cleaner code
    denom = torch.sin(omega) + 1e-8
    
    # 1. The warped functions
    f = torch.sin((1.0 - t_warped) * omega) / denom
    g = torch.sin(t_warped * omega) / denom
    
    # 2. The new derivatives (incorporating the chain rule via t_deriv)
    f_deriv = (-omega * torch.cos((1.0 - t_warped) * omega) / denom) * t_deriv
    g_deriv = (omega * torch.cos(t_warped * omega) / denom) * t_deriv
    
    return f, g, f_deriv, g_deriv

def _make_latent_sampler(name: str, *, device: torch.device, args: Optional[object] = None) -> Callable:
    lname = name.lower()

    if lname in {"gauss", "gaussian", "normal"}:
        def _sample(shape): 
            return torch.randn(*shape, device=device)
        return _sample

    if lname in {"uniform", "uni"}:
        def _sample(shape): return torch.rand(*shape, device=device) * 4.0 - 2.0
        return _sample

    if lname in {"student_t", "student-t", "studentt"}:
        default_dtype = torch.get_default_dtype()

        def _coerce_param(value, fallback):
            if value is None:
                return fallback
            if isinstance(value, str):
                text = value.strip()
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = float(text)
                return parsed
            return value

        df_param = _coerce_param(getattr(args, "student_t_df", None) if args is not None else None, 4.0)
        scale_param = _coerce_param(getattr(args, "student_t_scale", None) if args is not None else None, 1.0)

        df_tensor = torch.as_tensor(df_param, dtype=default_dtype, device=device)
        scale_tensor = torch.as_tensor(scale_param, dtype=default_dtype, device=device)
        loc_tensor = torch.zeros_like(df_tensor, dtype=default_dtype, device=device)

        dist = torch.distributions.StudentT(df=df_tensor, loc=loc_tensor, scale=scale_tensor)
        batch_shape = dist.batch_shape  # usually () or (dim,)

        def _sample(shape):
            if not shape:
                sample_shape = torch.Size()
            elif len(batch_shape) == 0:
                sample_shape = torch.Size(shape)
            else:
                if len(shape) < len(batch_shape):
                    raise ValueError(
                        "Requested Student-t sample shape is too small for batch parameters: "
                        f"shape={shape}, batch_shape={tuple(batch_shape)}"
                    )
                expected = tuple(batch_shape)
                actual = tuple(shape[-len(batch_shape):])
                if actual != expected:
                    raise ValueError(
                        "Student-t latent requires the trailing dimensions to match the parameter shape: "
                        f"expected {expected}, got {shape}"
                    )
                sample_shape = torch.Size(shape[:-len(batch_shape)])

            samples = dist.sample(sample_shape)
            if isinstance(samples, torch.Tensor):
                return samples
            return torch.as_tensor(samples, dtype=default_dtype, device=device)

        return _sample

  
    raise ValueError(f"Unknown baseline latent '{name}'")



def train_fm_baseline(
    args: SimpleNamespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the baseline flow-matching model."""
    device = torch.device(args.device)
    seed_all(args.seed)

    sampler = smpl.get_distribution(args.target_dataset)

    warmup_steps = max(0, int(getattr(args, "warmup_lr", 0)))

    def _warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / warmup_steps)

    # --- FIX: Only create scheduler here if optimizer was passed in ---
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
    else:
        scheduler = None 
    # ------------------------------------------------------------------

    flow_type = getattr(args, "baseline_flow", "linear").lower()
    flow_T = float(getattr(args, "baseline_flow_T", 1.0))
    if flow_T <= 0.0:
        raise ValueError("baseline_flow_T must be positive.")

    use_minibatch_ot = bool(getattr(args, "use_minibatch_ot", False))

    latent_sampler_train: Optional[Callable[[tuple[int, ...]], torch.Tensor]] = None
    latent_sampler_eval: Callable[[tuple[int, ...]], torch.Tensor]
    mmd_params: Optional[dict[str, object]] = None
    kac_params: Optional[dict[str, object]] = None

    if flow_type in {"linear", "wiener"}:
        latent_sampler_train = _make_latent_sampler(args.baseline_latent, device=device, args=args)

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            assert latent_sampler_train is not None
            return latent_sampler_train(shape)

        latent_sampler_eval = _latent_eval

    elif flow_type == "mmd":
        mmd_b = float(getattr(args, "baseline_mmd_b", 1.0))
        if mmd_b <= 0.0:
            raise ValueError("baseline_mmd_b must be positive.")
        mmd_sampler = smpl.TorchQuantileSampler(b=mmd_b, device=device, dtype=torch.float32)

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            batch = int(shape[0]) if len(shape) > 0 else 1
            t_final = torch.full((batch,), flow_T, device=device)
            latents, _ = mmd_sampler.sample(t_final, dim=args.dim)
            return latents.view(batch, -1)

        latent_sampler_eval = _latent_eval
        mmd_params = {"b": mmd_b, "sampler": mmd_sampler}

    elif flow_type == "kac":
        if use_minibatch_ot:
            print("[baseline] Disabling minibatch OT pairing for baseline_flow='kac'.")
            use_minibatch_ot = False
        kac_a = float(getattr(args, "baseline_kac_a", 9.0))
        kac_c = float(getattr(args, "baseline_kac_c", 3.0))
        kac_eps = float(getattr(args, "baseline_kac_epsilon", 1e-6))
        kac_M = int(getattr(args, "baseline_kac_lookup_M", 5000))
        kac_K = int(getattr(args, "baseline_kac_lookup_K", 1024))
        kac_sampler = smpl.TorchKacConstantSampler(
            a=kac_a,
            c=kac_c,
            T=flow_T,
            M=kac_M,
            K=kac_K,
            device=device,
            dtype=torch.float32,
        )

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            batch = int(shape[0]) if len(shape) > 0 else 1
            t_final = torch.full((batch,), flow_T, device=device)
            kac_1d = kac_sampler.sample(t_final, dim=1)
            uni = torch.randn(size = (batch, args.dim), device = device)
            latents = kac_1d * torch.nn.functional.normalize(uni)
            return latents.view(batch, -1)

        latent_sampler_eval = _latent_eval
        kac_params = {
            "a": kac_a,
            "c": kac_c,
            "epsilon": kac_eps,
            "sampler": kac_sampler,
        }
    elif flow_type == "target_norm":
        print("[baseline] Using direct target-norm latent sampler.")
        def _latent_eval(shape: tuple[int, ...], x0_batch) -> torch.Tensor:
            batch = int(shape[0]) if len(shape) > 0 else 1
            uni = torch.randn(size = (batch, args.dim), device = device)
            x0_norm = torch.linalg.vector_norm(x0_batch, dim=1).reshape(-1,1)
            latents = torch.nn.functional.normalize(uni) * x0_norm
            return latents.view(batch, -1)
        latent_sampler_eval = _latent_eval

    elif flow_type in {"target_norm_emp", "target_norm_interp"}:
        sampler = sampler.to(device)
        
        # --- DER FIX ---
        # Wenn der Sampler echte Daten im RAM hat (wie unser PIV), nimm einfach alle.
        # Wenn nicht (wie Swiss Roll), reichen 50.000 Samples für eine perfekte Verteilung.
        if hasattr(sampler, 'data'):
            n_eval_samples = sampler.data.shape[0] # Das sind dann deine ~665 echten PIV-Felder
            print(f"[baseline] Sampler has {n_eval_samples} real samples in RAM. Using all for norm estimation.")
        else:
            n_eval_samples = 25000 
            print(f"[baseline] Sampler does not have real samples in RAM. Using {n_eval_samples} samples for norm estimation.")
            
        print(f"Sampling {n_eval_samples} points for target norm estimation...")
        
        training_data = sampler.sample(n_eval_samples, device=device, dtype=torch.float32)
        
        print(f"X Min: {training_data[:, 0].min():.4f}, X Max: {training_data[:, 0].max():.4f}")
        print(f"Y Min: {training_data[:, 1].min():.4f}, Y Max: {training_data[:, 1].max():.4f}")
        """plot_neals_funnel(
            training_data[:, 1].cpu().numpy(),
            training_data[:, 0].cpu().numpy(),
            save_path=os.path.join(args.runs_dir, "funnel_plot_emp_sampling.png")
        )"""
        
        # Plot and log norm distribution with multiple views
        norms = torch.linalg.vector_norm(training_data, dim=1).cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Standard histogram
        axes[0, 0].hist(norms, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].set_xlabel('Norm')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Histogram (Linear Scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram with log-scale y-axis (to see outliers)
        axes[0, 1].hist(norms, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 1].set_xlabel('Norm')
        axes[0, 1].set_ylabel('Count (log)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Histogram (Log Scale Y)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram with log-scale both axes
        axes[0, 2].hist(norms, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 2].set_xlabel('Norm (log)')
        axes[0, 2].set_ylabel('Count (log)')
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_title('Histogram (Log-Log Scale)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. CDF - full range
        sorted_norms = np.sort(norms)
        cumulative = np.arange(1, len(sorted_norms) + 1) / len(sorted_norms)
        axes[1, 0].plot(sorted_norms, cumulative, linewidth=2, color='darkgreen')
        axes[1, 0].set_xlabel('Norm')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_title('Cumulative Distribution (Full)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. CDF - zoom on lower tail (outliers)
        p_low = 0.01  # 1% (outliers)
        norm_low = np.percentile(sorted_norms, p_low * 100)
        mask_low = sorted_norms <= norm_low
        axes[1, 1].plot(sorted_norms[mask_low], cumulative[mask_low], linewidth=2, color='darkred')
        axes[1, 1].set_xlabel('Norm')
        axes[1, 1].set_ylabel('CDF')
        axes[1, 1].set_title(f'CDF - Lower Tail (bottom {p_low*100:.1f}%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. CDF - zoom on upper tail (outliers)
        p_high = 0.99  # top 1% (outliers)
        norm_high = np.percentile(sorted_norms, p_high * 100)
        mask_high = sorted_norms >= norm_high
        axes[1, 2].plot(sorted_norms[mask_high], cumulative[mask_high], linewidth=2, color='darkorange')
        axes[1, 2].set_xlabel('Norm')
        axes[1, 2].set_ylabel('CDF')
        axes[1, 2].set_title(f'CDF - Upper Tail (top {(1-p_high)*100:.1f}%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"training_data/norm_distribution": wandb.Image(fig)}, step=0)
        plt.close(fig)
        
        # Print detailed statistics
        percentiles = [0.1, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        print(f"[baseline] Norm statistics:")
        print(f"  Overall: mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")
        print(f"  Percentiles:")
        for p in percentiles:
            val = np.percentile(norms, p)
            print(f"    {p:6.1f}%: {val:.4f}")
        
        if flow_type == "target_norm_emp":
            print("[baseline] Using empirical latent norm sampler.")
            empNormSampler = smpl.EmpiricalNormSampler(training_data)
            latent_sampler_train = empNormSampler.sample_norms
            get_heavy_tail_samples = empNormSampler.get_heavy_tail_samples
        else:
            print("[baseline] Using interpolated latent norm sampler.")
            latent_sampler_train = smpl.InterpolatedNormSampler(training_data).sample_norms

        def _latent_eval(shape: tuple[int,...], seed_offset: int = 999) -> torch.Tensor:
            """Generate latent noise with fixed seed for reproducible evaluation.
            
            Args:
                shape: (batch_size, dim) shape
                seed_offset: seed offset to use (default 999 for eval)
            """
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + seed_offset)
            uni = torch.randn(size=shape, device=device, generator=gen)
            uni = torch.nn.functional.normalize(uni)
            return latent_sampler_train(shape[0]).reshape(-1,1) * uni
        latent_sampler_eval = _latent_eval
            
    else:
        raise ValueError(f"Unsupported baseline_flow '{flow_type}'")

    if getattr(args, "is_msgm", False):
        print("--- INITIALIZING MSGM ARCHITECTURE ---")
        from learn_noise.msgm_lib.NN import MLP as MSGM_MLP
        from learn_noise.msgm_lib.SDEs import multiplicativeNoise, PluginReverseSDE
        
        # 1. Initialize MSGM Network
        # FIXED:
        drift_q = MSGM_MLP(input_dim=args.dim, index_dim=1, hidden_dim=args.hidden_size, premodule="NormalizeLogRadius").to(device)
        T_val = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)
        
        # 2. Initialize SDE Base (using a dummy batch to setup shapes)
        dummy_x = sampler.sample(1000, device=device, dtype=torch.float32)
        inf_sde = multiplicativeNoise(
            dummy_x, beta_min=args.msgm_beta_min, beta_max=args.msgm_beta_max,
            t_epsilon=args.msgm_t_eps, T=T_val, num_steps_forward=args.msgm_steps, device=device
        )
        
        # 3. Compile full MSGM Model
        model = PluginReverseSDE(inf_sde, drift_q, T_val, vtype='rademacher', debias=False, ssm_intT=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
        
        # 4. Create spoof wrappers for your plotting/eval functions!
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
        wrapper = MSGMSpoofWrapper(ema, device)
        ode_func = wrapper # Point the ODE solver directly to the spoof!

        # =================================================================
        # --- THE SDE MONKEY PATCH (HEUN + SINGULARITY AVOIDANCE) ---
        # =================================================================
        def msgm_sde_odeint(func, y0, t, **kwargs):
            # Switch to Heun for 2nd-order curve-hugging on the sphere!
            from learn_noise.msgm_lib.sde_scheme import heun_sampler
            
            raw_msgm_model = func._base_model if hasattr(func, '_base_model') else func
            needs_correction = getattr(raw_msgm_model.base_sde, 'norm_correction', False)
            
            # 1. CRITICAL: Prevent Score Explosion!
            # We must stop integrating right before the score blows up to infinity.
            # Reverse SDE time goes from T down to 0. We stop at t_epsilon.
            # 1. CRITICAL: Prevent Score Explosion!
            # Calculate the safe float, then wrap it in a Tensor so Heun can call .item() on it.
            safe_t_val = raw_msgm_model.T.item() - raw_msgm_model.base_sde.t_epsilon
            safe_T = torch.tensor([safe_t_val], device=y0.device)
            
            print(f"\n[MSGM SDE] Safe T: {safe_t_val:.4f} | Norm Correct: {needs_correction}")
            
            with torch.no_grad():
                # 2. Use Heun (RK2) and 256 steps for stable convergence
                xs_list = heun_sampler(
                    raw_msgm_model, 
                    y0, 
                    num_steps=256, 
                    keep_all_samples=True,
                    norm_correction=needs_correction,
                    T_=safe_T # <-- Stops the whirlpool slingshot!
                )
            
            if isinstance(xs_list, list):
                xs = torch.stack(xs_list, dim=0)
            else:
                xs = xs_list
                
            movement = (xs[-1] - xs[0]).abs().mean().item()
            print(f"[MSGM SDE] Integration complete! Average movement: {movement:.6f}")
                
            indices = torch.linspace(0, len(xs) - 1, len(t)).long()
            return xs[indices]
        # 1. Patch the global torchdiffeq module (just in case)
        import torchdiffeq
        torchdiffeq.odeint = msgm_sde_odeint

        # 2. THE CRITICAL FIX: Patch the local references in your actual files!
        import learn_noise.utils.plotting_traj
        import learn_noise.utils.evaluation
        import learn_noise.training.logging
        import learn_noise.training.sample_utils
        
        # This forces your evaluation scripts to use the SDE solver instead of the ODE solver
        learn_noise.utils.plotting_traj.odeint = msgm_sde_odeint
        learn_noise.utils.evaluation.odeint = msgm_sde_odeint
        learn_noise.training.logging.odeint = msgm_sde_odeint
        learn_noise.training.sample_utils.odeint = msgm_sde_odeint
        # =================================================================
        
    else:
        # --- YOUR ORIGINAL SETUP CODE GOES HERE ---
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
        if args.slerp:
            ang_mode = AngularInferenceWrapper(ema)
            wrapper = TorchWrapper(ang_mode)
        else:
            wrapper = TorchWrapper(ema)
        ode_func = ODEWrapper(wrapper).to(device)
    
    unet_params = count_parameters(model)
    model_size_stats = {
        "method": "baseline_fm",
        "target_dataset": getattr(args, "target_dataset", None),
        "params_unet": unet_params,
        "params_quantile": 0,
        "params_total": unet_params,
    }
    channel_mult = getattr(args, "unet_channel_mult", None)
    if channel_mult is not None:
        model_size_stats["unet_channel_mult"] = tuple(channel_mult)
    for attr in ("unet_model_channels", "unet_num_res_blocks", "unet_attention_resolutions"):
        value = getattr(args, attr, None)
        if value is not None:
            model_size_stats[attr] = value
    write_model_size_summary(args.runs_dir, model_size_stats)
    wandb.log({
        "params/unet": float(unet_params),
        "params/quantile": 0.0,
        "params/total": float(unet_params),
    }, step=0)

    fixed_sampler = make_fixed_sampler(sampler, seed=args.seed, device=device)
    if not hasattr(args, "_fixed_baseline_x0"):
        args._fixed_baseline_x0 = fixed_sampler(args.batch_size, seed_offset=0)
    x0_fixed = args._fixed_baseline_x0

    image_shape = getattr(args, "image_shape", None)

    image_shape = np.array([32,32])

    image_dim = math.prod(image_shape) if image_shape is not None else None



    is_image_task = image_shape is not None and image_dim == args.dim


    if is_image_task:
        log_real_rgb_histogram_once(
            args=args,
            sampler=sampler,
            image_shape=image_shape,
            device=device,
            step=0,
        )

    checkpoint_dir = os.path.join(args.runs_dir, "baseline_fm")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fid_interval = int(args.fid_eval_interval) if hasattr(args, "fid_eval_interval") else 0
    fid_num_gen = int(args.fid_num_gen) if hasattr(args, "fid_num_gen") else 0
    fid_batch_size = max(1, int(getattr(args, "fid_batch_size", args.batch_size))) if fid_interval > 0 else 0
    fid_gen_batch = max(1, int(getattr(args, "fid_gen_batch", args.batch_size))) if fid_interval > 0 else 0
    fid_image_size = (
        int(getattr(args, "fid_image_size", 0)) if (fid_interval > 0 and image_shape is not None) else 0
    )
    fid_real_cache = None
    if is_image_task and fid_interval > 0 and fid_num_gen > 0:
        with torch.no_grad():
            real_samples = sampler.sample(fid_num_gen, device=device, dtype=torch.float32)
            fid_real_cache = reshape_flat_samples(real_samples, torch.Size(image_shape)).detach().cpu()

    sample_vis_interval = int(getattr(args, "sample_vis_interval", 0))
    sample_vis_count = int(getattr(args, "sample_vis_count", 0))
    sample_vis_nrow = int(getattr(args, "sample_vis_nrow", 8))

    sample_dir = os.path.join(checkpoint_dir, "samples") if is_image_task else ""
    t_eval = torch.linspace(1.0, 0.0, args.num_steps_eval, device=device)

    fixed_vis_noise = getattr(args, "_fixed_baseline_vis_noise", None) if is_image_task else None

    train_time_accumulator = 0.0

    strat_ot = getattr(args, "strat_ot", False)

    if strat_ot:
        x_data = sampler.sample(args.batch_size * args.epochs, device=device, dtype=torch.float32)
        N = len(x_data)
        x_data = x_data * torch.log1p(x_data.norm(dim=1, keepdim=True))/(x_data.norm(dim=1, keepdim=True)+1e-8) 

        # =========================================================================
        # NEU: 1. Relative Outlier Augmentation (Vektorisiert & Schnell)
        # =========================================================================
        use_rel_outlier_aug = False
        r_orig = torch.linalg.norm(x_data, dim=1)
        if use_rel_outlier_aug:
            print("1a. Dichte-bewusste (Density-Aware) stetige OT-Glättung...")
            r_max = r_orig.max() + 1e-8
            
            # =========================================================
            # 1. LOKALE DICHTE MESSEN (Der Sparsity-Sensor)
            # =========================================================
            # Wir sortieren die Radien, um zu sehen, wie weit die Nachbarn entfernt sind
            r_sorted, sort_idx = torch.sort(r_orig)
            
            # Fenstergröße: Wie viele Nachbarn schauen wir uns an? (z.B. 50)
            window = 50 
            pad_size = window // 2
            
            # Ränder auffüllen für saubere Berechnung
            r_padded = torch.nn.functional.pad(r_sorted.unsqueeze(0), (pad_size, pad_size), mode='replicate').squeeze(0)
            
            # "Sparsity" (Leere): Abstand zwischen dem rechten und linken Rand des Nachbarschafts-Fensters
            # Großer Wert = Punkt ist sehr einsam (Graben) -> Braucht Klone!
            # Kleiner Wert = Punkt ist in einer Wolke (Klumpen) -> Braucht weniger Klone!
            sparsity_sorted = r_padded[window:] - r_padded[:-window]
            
            # Um zu wissen, was ein "Graben" ist, berechnen wir den gleitenden Durchschnitt (Trend)
            kernel_size = min(501, max(3, N // 50))
            if kernel_size % 2 == 0: kernel_size += 1 # Muss ungerade sein
            
            trend_sparsity = torch.nn.functional.avg_pool1d(
                sparsity_sorted.view(1, 1, -1), 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size // 2
            ).view(-1)
            
            # Multiplikator: Lokale Leere geteilt durch den Trend
            # > 1.0 bedeutet: Hier ist ein Loch!
            # < 1.0 bedeutet: Hier ist ein Klumpen!
            sparsity_multiplier_sorted = sparsity_sorted / (trend_sparsity + 1e-8)
            
            # Limitiere den Effekt, damit es nicht explodiert (z.B. max 3-facher Boost)
            sparsity_multiplier_sorted = torch.clamp(sparsity_multiplier_sorted, 0.2, 2.0)
            
            # Werte wieder den originalen, unsortierten Punkten zuweisen
            unsort_idx = torch.argsort(sort_idx)
            sparsity_multiplier = sparsity_multiplier_sorted[unsort_idx]
            local_sparsity = sparsity_sorted[unsort_idx] # Merken wir uns für das Rauschen!

            # =========================================================
            # 2. DIE EXPONENTIELLE BASIS-KURVE
            # =========================================================
            max_clones = 700.0  
            k = 9.0             
            r_norm = r_orig / r_max 
            
            base_expected_clones = max_clones * (torch.exp(k * r_norm) - 1.0) / (math.exp(k) - 1.0)
            
            # =========================================================
            # 3. DIE MAGIE: Basis-Wachstum mal Dichte-Boost
            # =========================================================
            # Ein Punkt weiter innen im Graben kann jetzt MEHR Klone bekommen 
            # als ein Punkt weiter außen im Klumpen!
            expected_clones = base_expected_clones * sparsity_multiplier
            
            # Stochastisches Runden
            base_clones = torch.floor(expected_clones).int()
            prob_extra = expected_clones - base_clones.float()
            extra_clone = torch.bernoulli(prob_extra).int() 
            
            num_clones = base_clones + extra_clone

            
            
            # =========================================================
            # 4. KLONEN & ADAPTIVES RAUSCHEN
            # =========================================================
            mask = num_clones > 0
            if mask.any():
                outlier_points = x_data[mask]
                repeats = num_clones[mask]
                
                clones = torch.repeat_interleave(outlier_points, repeats, dim=0)
                
                # DER ZWEITE MAGIC TRICK: "Gap-Filling Noise"
                # Statt pauschal 1% Radius-Rauschen, machen wir das Rauschen genau so groß
                # wie die lokale Lücke (Sparsity) an diesem Ort!
                # Das schmiert die Klone exakt über den leeren Raum zwischen den Datenpunkten!
                point_gaps = torch.repeat_interleave(local_sparsity[mask], repeats, dim=0).unsqueeze(1)
                
                # Faktor 0.3 bis 0.5 der Lückengröße erzeugt perfekte, nahtlose Übergänge
                noise_scale = 0.4 * point_gaps 
                noise = torch.randn_like(clones) * noise_scale
                
                clones_noisy = clones + noise
                
                x_data_aug = torch.cat([x_data, clones_noisy], dim=0)
                print(f" -> {len(clones)} smarte Gap-Filler Klone erzeugt.")
            else:
                x_data_aug = x_data

            N_aug = len(x_data_aug)
            print(f" -> Neues Dataset-Volumen: {N_aug} Punkte.")
            print("-> Erstelle Verteilungs-Plot zur Kontrolle...")

            # 1. Daten für den Plot vorbereiten (auf CPU holen)
            # Wir vergleichen das Original (r_orig) mit dem Ergebnis (x_data_aug)
            r_np_orig = r_orig.detach().cpu().numpy()
            r_np_aug = torch.linalg.norm(x_data_aug, dim=1).detach().cpu().numpy()
            
            # 2. Feines Histogramm berechnen (als Basis für die Kurve)
            # Wichtig: "density=True", damit die Kurven vergleichbar sind, auch wenn N unterschiedlich ist
            bins = 500
            range_max = r_np_aug.max()
            hist_orig, bin_edges = np.histogram(r_np_orig, bins=bins, range=(0, range_max), density=True)
            hist_aug, _ = np.histogram(r_np_aug, bins=bins, range=(0, range_max), density=True)
            
            # Die X-Achse (Mittelpunkte der Bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 3. Glättungs-Funktion (Moving Average) für die "schöne Linie"
            def smooth_curve(y, box_pts=10):
                box = np.ones(box_pts) / box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth

            # Wir glätten leicht, um das Rauschen des Histogramms zu entfernen
            curve_orig = smooth_curve(hist_orig, box_pts=5)
            curve_aug = smooth_curve(hist_aug, box_pts=5)

            # 4. Der Plot
            plt.figure(figsize=(10, 6))
            
            # Original Kurve (Blau, gestrichelt)
            plt.plot(bin_centers, curve_orig, 'b--', label='Original Verteilung (Referenz)', alpha=0.6, linewidth=1.5)
            
            # Augmentierte Kurve (Rot, durchgezogen)
            plt.plot(bin_centers, curve_aug, 'r-', label='Mit Exponentieller Augmentierung', alpha=0.8, linewidth=2)
            
            plt.yscale('log') # WICHTIG: Log-Skala, um die Tails zu sehen!
            plt.xlabel('Radius (Norm)')
            plt.ylabel('Log-Dichte (Häufigkeit)')
            plt.title(f'Kontrolle der Monotonie (k={k}, max_clones={max_clones})')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()
            
            # Zoom-In Region markieren (optional, wo die Action passiert)
            p99_visual = np.percentile(r_np_orig, 99)
            plt.axvline(x=p99_visual, color='k', linestyle=':', alpha=0.5, label='Top 1% (Visuelle Orientierung)')

            plt.tight_layout()
            plt.savefig('distribution_check.png', dpi=150)
            plt.savefig(os.path.join(args.runs_dir, "distribution_check.png"), dpi=150)
            plt.close()
        else:
            x_data_aug = x_data
            N_aug = N
            print("1a. Keine Augmentierung angewendet, verwende Originaldaten.")
        # =========================================================================
        # 2. Polarkoordinaten & Sortierung (Jetzt mit dem erweiterten Dataset!)
        # =========================================================================
        print("1b. Berechne Polarkoordinaten und sortiere nach Radius...")
        r_data = torch.linalg.norm(x_data_aug, dim=1)
        phi_data = torch.atan2(x_data_aug[:, 1], x_data_aug[:, 0])

        sorted_r, sort_idx = torch.sort(r_data)
        sorted_phi = phi_data[sort_idx]
        sorted_x_target = x_data_aug[sort_idx]

        print("2. Führe adaptives Multi-Shift Circular OT durch...")
        noise_angles = torch.empty_like(sorted_phi)
        tracked_radii = []
        tracked_seam_angles = []

        # =====================================================================
        # METHODEN-SWITCH: Welche 1D-OT Methode soll im "Bulk" verwendet werden?
        # =====================================================================
        use_exact_1d_circular_ot = True  # True = Neue exakte Methode, False = Alte Heuristik
        
        last_cut_angle = 0.0 
        start_idx = 0
        
        while start_idx < N_aug:
            progress = start_idx / N_aug
            
            # DIE BEDINGUNG: Wann 1D und wann 2D? (z.B. Bulk = 1D, Tails = 2D)
            is_bulk = progress < 0.90
            
            # Chunk-Größe dynamisch anpassen
            current_chunk_size = 5000 if is_bulk else 200
            end_idx = min(start_idx + current_chunk_size, N_aug)
            actual_chunk_size = end_idx - start_idx
            
            chunk_phi_target_raw = sorted_phi[start_idx:end_idx]
            
            if is_bulk and use_exact_1d_circular_ot:
                # -----------------------------------------------------------------
                # NEU: Exaktes 1D Circular Optimal Transport (Dein neuer Standard)
                # -----------------------------------------------------------------
                chunk_phi_noise = torch.linspace(
                    -math.pi, 
                    math.pi - (2 * math.pi / actual_chunk_size), 
                    actual_chunk_size, 
                    device=device
                )
                
                alpha_sorted, idx_alpha = torch.sort(chunk_phi_target_raw)
                beta_sorted = chunk_phi_noise
                
                K = actual_chunk_size
                beta_expanded = beta_sorted.repeat(2)
                shifts = beta_expanded.unfold(0, K, 1)[:K] 
                
                diff = torch.abs(alpha_sorted.unsqueeze(0) - shifts)
                circ_dist = torch.minimum(diff, 2*math.pi - diff)
                
                costs = circ_dist.sum(dim=1)
                best_shift_idx = torch.argmin(costs)
                
                best_beta = shifts[best_shift_idx]
                
                final_noise_phi = torch.empty_like(best_beta)
                final_noise_phi[idx_alpha] = best_beta
                
                noise_angles[start_idx:end_idx] = final_noise_phi
                
                tracked_radii.append(sorted_r[start_idx:end_idx].mean().item())
                tracked_seam_angles.append(best_beta[0].item()) # Startpunkt des Rauschens als "Seam"

            elif is_bulk and not use_exact_1d_circular_ot:
                # -----------------------------------------------------------------
                # ALT: Deine Center-of-Mass Heuristik (als Fallback)
                # -----------------------------------------------------------------
                chunk_phi_noise = torch.linspace(
                    -math.pi, 
                    math.pi - (2 * math.pi / actual_chunk_size), 
                    actual_chunk_size, 
                    device=device
                )
                
                sin_mean = torch.sin(chunk_phi_target_raw).mean()
                cos_mean = torch.cos(chunk_phi_target_raw).mean()
                r_mean = torch.hypot(sin_mean, cos_mean) 
                
                center_of_mass_angle = torch.atan2(sin_mean, cos_mean)
                target_cut_angle = (center_of_mass_angle + math.pi) % (2 * math.pi) - math.pi
                
                if r_mean < 0.2:
                    cut_angle = last_cut_angle
                else:
                    diff = target_cut_angle - last_cut_angle
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    if abs(diff) < (math.pi / 2):
                        cut_angle = last_cut_angle + (diff * 0.5) 
                    else:
                        cut_angle = target_cut_angle 
                        
                cut_angle = (cut_angle + math.pi) % (2 * math.pi) - math.pi
                last_cut_angle = cut_angle
                
                phi_target_shifted = (chunk_phi_target_raw - last_cut_angle + math.pi) % (2 * math.pi) - math.pi
                _, target_sort_idx = torch.sort(phi_target_shifted)
                
                aligned_noise_phi = torch.empty_like(chunk_phi_noise)
                aligned_noise_phi[target_sort_idx] = chunk_phi_noise
                
                final_noise_phi = (aligned_noise_phi + last_cut_angle + math.pi) % (2 * math.pi) - math.pi
                
                noise_angles[start_idx:end_idx] = final_noise_phi
                
                tracked_radii.append(sorted_r[start_idx:end_idx].mean().item())
                tracked_seam_angles.append(cut_angle)

            else:
                # -----------------------------------------------------------------
                # TAILS: Echtes 2D Optimal Transport (Für die letzten 10%)
                # -----------------------------------------------------------------
                chunk_phi_noise = torch.linspace(
                    -math.pi, 
                    math.pi - (2 * math.pi / actual_chunk_size), 
                    actual_chunk_size, 
                    device=device
                )
                
                r_chunk = sorted_r[start_idx:end_idx]
                target_x = sorted_x_target[start_idx:end_idx] 
                
                noise_x = torch.stack([
                    r_chunk * torch.cos(chunk_phi_noise),
                    r_chunk * torch.sin(chunk_phi_noise)
                ], dim=1)
                
                cost_matrix = torch.cdist(target_x, noise_x, p=2).cpu().numpy()
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                
                matched_noise_phi = chunk_phi_noise[col_ind]
                noise_angles[start_idx:end_idx] = matched_noise_phi
                
                tracked_radii.append(r_chunk.mean().item())
                tracked_seam_angles.append(last_cut_angle)
                
            start_idx = end_idx

        print("3. Erstelle die gekoppelten Startpunkte (Noise)...")
        x_noise_start = torch.stack([
            sorted_r * torch.cos(noise_angles),
            sorted_r * torch.sin(noise_angles)
        ], dim=1)

        print("4. Mische die Paare für das Training (Verhindert Sortier-Bias)...")
        shuffle_idx = torch.randperm(N_aug, device=device) # ACHTUNG: Hier jetzt N_aug nutzen!
        x_target_shuffled = sorted_x_target[shuffle_idx]
        x_noise_shuffled = x_noise_start[shuffle_idx]


        print("5. Optischer Sanity-Check: Plotte eine Stichprobe der OT-Trajektorien...")
        num_plot = 500  # 500 Punkte sind perfekt für einen sauberen Plot

        # Hole die Tensoren auf die CPU und wandle sie für Matplotlib in Numpy-Arrays um
        plot_noise = x_noise_shuffled[:num_plot].cpu().numpy()
        plot_target = x_target_shuffled[:num_plot].cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        plt.title(f"Lokales 2D Optimal Transport (Zeige {num_plot} Paare)")

        # 1. Zeichne die Verbindungslinien (die Wege, die dein Netz lernen muss)
        for i in range(num_plot):
            plt.plot([plot_noise[i, 0], plot_target[i, 0]], 
                    [plot_noise[i, 1], plot_target[i, 1]], 
                    color='gray', alpha=0.4, linewidth=0.8, zorder=1)

        # 2. Zeichne die Startpunkte (Uniform Noise)
        plt.scatter(plot_noise[:, 0], plot_noise[:, 1], 
                    c='royalblue', s=15, label='Start (Noise)', zorder=2)

        # 3. Zeichne die Zielpunkte (Echte Funnel-Daten)
        plt.scatter(plot_target[:, 0], plot_target[:, 1], 
                    c='darkorange', s=15, label='Ziel (Target)', zorder=3)

        plt.legend()
        plt.axis('equal') # Verhindert, dass die Kreise zu Ellipsen verzerrt werden
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.runs_dir, "ot_sanity_check.png"), dpi=300)
        plt.close(fig)

        # Wandle die getrackten Listen in Numpy-Arrays um
        seam_radii_np = np.array(tracked_radii)
        # Winkel bereinigen (auf -pi bis +pi mappen)
        seam_angles_np = (np.array(tracked_seam_angles) + np.pi) % (2 * np.pi) - np.pi

        # Polarkoordinaten der Schnittstellen in X/Y umrechnen
        seam_x = seam_radii_np * np.cos(seam_angles_np)
        seam_y = seam_radii_np * np.sin(seam_angles_np)

        fig = plt.figure(figsize=(10, 10))
        plt.title("Evolution der topologischen Schnittstelle über alle Ringe")

        # 1. Echte Daten im Hintergrund (Stichprobe von 50.000 Punkten für Performance)
        bg_plot_size = min(N, 50000)
        bg_x = sorted_x_target[:bg_plot_size, 0].cpu().numpy()
        bg_y = sorted_x_target[:bg_plot_size, 1].cpu().numpy()

        plt.scatter(bg_x, bg_y, s=2, color='lightblue', alpha=0.3, label='Target Blobs', zorder=1)

        # 2. Plotte die Schnittstelle als Linie und Punkte
        plt.plot(seam_x, seam_y, color='red', linewidth=1.5, alpha=0.7, label='Cut-Verlauf', zorder=2)
        plt.scatter(seam_x, seam_y, color='darkred', s=15, edgecolor='white', linewidth=0.5, zorder=3)

        # 3. Markiere den Start (tiefster Ring) und das Ende (äußerster Ring)
        plt.scatter(seam_x[0], seam_y[0], color='lime', s=80, marker='*', edgecolor='black', label='Start (Zentrum)', zorder=4)
        plt.scatter(seam_x[-1], seam_y[-1], color='magenta', s=80, marker='X', edgecolor='black', label='Ende (Außen)', zorder=4)

        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(args.runs_dir, "seam_evolution.png"), dpi=300)
        plt.close(fig)

    print("Starte Trainingsloop...")


    
    for step in tqdm(range(args.epochs), desc="Flow-matching baseline"):
        start_idx = step * args.batch_size
        end_idx = start_idx + args.batch_size
        iter_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pairing_cost = None
        if strat_ot:
            x_0 = x_target_shuffled[start_idx:end_idx].to(device)#sampler.sample(args.batch_size, device=device, dtype=torch.float32)
        else:
            x_0 = sampler.sample(args.batch_size, device=device, dtype=torch.float32)

        if getattr(args, "is_msgm", False):
            # ==========================================
            # MSGM SCORE MATCHING LOOP
            # ==========================================
            # MSGM calculates the noise, time sampling, and loss internally!
            loss_mse = model.ssm(x_0).mean()
            loss = loss_mse
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.model_grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)

        else: 
            if x_0.dim() > 2:
                x_0 = x_0.view(x_0.shape[0], -1)
            
            t = torch.rand(args.batch_size, 1, device=device)
            t_actual = t.squeeze(1) * flow_T

            if flow_type == "linear":
                assert latent_sampler_train is not None
                z = latent_sampler_train((args.batch_size, args.dim))
                if use_minibatch_ot:
                    idx_best, transport_plan = minibatch_ot_pairing(x_0, z)
                    x_0 = x_0[idx_best]
            
                    pairing_cost = transport_plan.max(dim=0).values.mean()
                x_t = (1.0 - t) * x_0 + t * z
                velocity_target = -x_0 + z

            elif flow_type == "mmd":
                assert mmd_params is not None
                tau_raw, U = mmd_params["sampler"].sample(t_actual, dim=args.dim)
                tau = tau_raw.view(args.batch_size, -1).to(device=device)
                U = U.view(args.batch_size, -1).to(device=device)
                disp = 2.0 * U - 1.0
                f = 1.0 - t
                x_t = f * x_0 + tau
                velocity_noise = compute_velocity_mmd(
                    x=disp,
                    t=t_actual.unsqueeze(1),
                    b=mmd_params["b"],
                    disp=disp,
                )
                velocity_noise = velocity_noise * flow_T
                velocity_target = -x_0 + velocity_noise

            elif flow_type == "kac":
                assert kac_params is not None
                tau_raw = kac_params["sampler"].sample(t_actual, dim=1)
                tau = tau_raw.view(args.batch_size, -1).to(device=device)
                f = 1.0 - t
                uni = torch.randn(size = (args.batch_size, args.dim), device = device)
                uni = torch.nn.functional.normalize(uni)
                x_t = f * x_0 + uni * tau
                x_t_cond0 = x_t - f * x_0
                x_t_cond0_norm = torch.linalg.vector_norm(x_t_cond0, dim=1)
                x_t_cond0_norm = x_t_cond0_norm[:,None]
                velocity_noise_rad = compute_velocity_kac(
                    x_t_cond0_norm,
                    t_actual.unsqueeze(1),
                    a=kac_params["a"],
                    c=kac_params["c"],
                    epsilon=kac_params["epsilon"],
                    T=flow_T,
                )
                velocity_target = -x_0 + velocity_noise_rad * torch.nn.functional.normalize(x_t_cond0)

            elif flow_type in {"target_norm", "target_norm_emp", "target_norm_interp"}:
                use_minibatch_ot = getattr(args, "use_minibatch_ot", False)

                f = 1.0 - t
                g = t
                f_deriv = -1.0
                g_deriv = 1.0
                uni = torch.randn(size = (args.batch_size, args.dim), device = device)
                uni = torch.nn.functional.normalize(uni)
                # For slerp, use x_0 norms directly (no need for sampled_norm)

                # tuning down to log1p size
                if not strat_ot:
                    uni = torch.randn(size = (args.batch_size, args.dim), device = device)
                    uni = torch.nn.functional.normalize(uni)
                    x_0_norm = torch.linalg.vector_norm(x_0, dim = 1).reshape(-1,1)
                    x_0_log1p = torch.log1p(x_0_norm).reshape(-1,1)
                    x_0 = x_0 / (x_0_norm + 1e-8) * x_0_log1p
                    x0_lognorms = torch.linalg.vector_norm(x_0, dim = 1)
                    noise = x0_lognorms.unsqueeze(1) * uni
                else: 
                    x0_lognorms = torch.linalg.vector_norm(x_0, dim = 1)
                    noise = x_noise_shuffled[start_idx:end_idx]

                if use_minibatch_ot:
                    idx_best, transport_plan = minibatch_ot_pairing(x_0, noise)
                    x_0 = x_0[idx_best]
                    #z = z[idx_best]
                    pairing_cost = transport_plan.max(dim=0).values.mean()

                elif args.slerp:
                    # Full radius matching: x_0 and noise are already perfectly aligned by construction
                    # (noise = x_0_norm * uni, so ||noise|| = ||x_0||)
                    # Compute angle omega between x_0 and noise using simplified formula
                    # Since ||x_0|| = ||noise||: cos(omega) = (x_0 · noise) / ||x_0||^2

                    pairing_cost = None

                    Radius = x0_lognorms ** 2
                    R_true = torch.sqrt(Radius + 1e-8)
                    if R_true.dim() == 1: R_true = R_true.unsqueeze(1) # Shape-Fix

                    dot_product = (x_0 * noise).sum(dim=1)  
                    cos_omega = dot_product / (Radius + 1e-8)
                    cos_omega = torch.clamp(cos_omega, -0.999, 0.999)  
                    omega = torch.acos(cos_omega).unsqueeze(1)

                    # 2. Den VOLLEN, echten Weg berechnen (WICHTIG!)
                    p_chosen = 1
                    if strat_ot:
                        p_chosen = 1
                    
                    f,g,f_deriv,g_deriv = get_slerp_warped(t, omega, p=p_chosen)

                else:
                    pairing_cost = None
                
                # Compute trajectory and velocity for all matching methods
                x_t = f * x_0 + g * noise

                # Hier berechnest du das ideale Vektorfeld (noch euklidisch/slerp)
                velocity_target = f_deriv * x_0 + g_deriv * noise
                #velocity_target = velocity_target / (x_t.norm(dim=1, keepdim=True) + 1e-8)   

            else:
                raise RuntimeError(f"Unexpected baseline_flow '{flow_type}' at training time")
    
            velocity_pred = model(t, x_t)

            loss_mse_log_perSample = F.mse_loss(velocity_pred, velocity_target, reduction = 'none')

            if args.slerp:
                radius = x_t.detach().norm(dim=1, keepdim=True)
                weight_radius = torch.expm1(radius)
                dist_to_seam = torch.abs(x_t[:, 1:2]) 
                weight_seam = 1.0 / (dist_to_seam + 0.02) 

                weights = weight_radius * weight_seam
                weights = weights / weights.mean()

                weights = radius * (2-t)
            else:
                weights = 1.0
            
            loss_mse = (loss_mse_log_perSample * weights**2).mean()

            loss = loss_mse
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.model_grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)

        train_time_accumulator += time.perf_counter() - iter_start

        if step % 100 == 0:
            log_payload = {
                'loss/velocity': float(loss.item()),
                'loss/velocity_mse': float(loss_mse.item()),
                'grad/model_velocity': float(grad_norm.item()),
            }
            if pairing_cost is not None:
                log_payload['metrics/minibatch_ot_cost'] = float(pairing_cost.item())
            wandb.log(log_payload, step=step)

        # =========================================================================
        # NEU: SANITY CHECK (Alle 1000 Schritte direkt im Terminal)
        # =========================================================================
        if step % 1000 == 0 and step > 0:
            with torch.no_grad():
                # Nimm das erste Sample aus dem aktuellen Batch
                v_pred_sample = velocity_pred[0].detach().cpu().numpy()
                v_targ_sample = velocity_target[0].detach().cpu().numpy()
                
                # Prüfe, ob es unsere 1024D (32x32) PIV-Daten sind
                if args.dim == 1024:
                    try:
                        # Zurück ins 32x32 Grid falten
                        v_pred_img = v_pred_sample.reshape((32, 32), order='F')
                        v_targ_img = v_targ_sample.reshape((32, 32), order='F')
                        
                        # Lokale Glätte (Gradient) berechnen
                        def calc_smoothness(img_2d):
                            diff_y = np.mean(np.abs(img_2d[1:, :] - img_2d[:-1, :]))
                            diff_x = np.mean(np.abs(img_2d[:, 1:] - img_2d[:, :-1]))
                            return (diff_x + diff_y) / 2.0
                            
                        sm_pred = calc_smoothness(v_pred_img)
                        sm_targ = calc_smoothness(v_targ_img)
                        
                        print(f"\n--- [SANITY CHECK | Step {step}] ---")
                        print(f"ZIEL (Target) -> Mean: {v_targ_sample.mean():.4f}, Std: {v_targ_sample.std():.4f}, Smoothness: {sm_targ:.4f}")
                        print(f"NETZ (Pred)   -> Mean: {v_pred_sample.mean():.4f}, Std: {v_pred_sample.std():.4f}, Smoothness: {sm_pred:.4f}")
                        
                        # Alarm, falls das Netz kollabiert ist!
                        if np.abs(v_pred_sample).max() < 1e-4:
                            print("!!! WARNUNG: Das U-Net gibt (fast) nur Nullen aus! Learning Rate zu hoch? !!!")
                        print("----------------------------------\n")
                    except Exception as e:
                        print(f"Sanity Check Error: {e}")
        # =========================================================================

        do_light = (args.eval_sample > 0) and (((step + 1) % args.eval_step) == 0)
        do_heavy = (args.big_eval_samples > 0) and (((step + 1) % args.big_eval_step) == 0)
        
        ###################### EVAL 2D ######################
        if not is_image_task and (do_light or do_heavy):
            if args.slerp:
                plotting.plot_vector_field(model = ema, device = device, t = 0.0, path = args.runs_dir, step = step)
                plotting.plot_vector_field(model = ema, device = device, t = 0.5, path = args.runs_dir, step = step)
                plotting.plot_vector_field(model = ema, device = device, t = 0.9, path = args.runs_dir, step = step)
                plotting.inspect_single_trajectory(model = ema, start_noise = torch.tensor([[300.0, 1.0]]), t_vals = t_eval, path = args.runs_dir, step = step)
                plotting.inspect_single_trajectory(model = ema, start_noise = torch.tensor([[300.0, 0.0]]), t_vals = t_eval, path = args.runs_dir, step = step)
                plotting.inspect_single_trajectory(model = ema, start_noise = torch.tensor([[300.0, 0.5]]), t_vals = t_eval, path = args.runs_dir, step = step)
                plotting.plot_dead_zone_velocity(model = ema, x_value = 50.0,device = device, path = args.runs_dir, step = step)
                plotting.plot_dead_zone_velocity(model = ema, x_value = 100.0,device = device, path = args.runs_dir, step = step)
                plotting.plot_dead_zone_velocity(model = ema, x_value = 200.0,device = device, path = args.runs_dir, step = step)
                plotting.plot_dead_zone_velocity(model = ema, x_value = 300.0,device = device, path = args.runs_dir, step = step)

            log_baseline_evaluation(
                args=args,
                step=step,
                ema_model=ema,
                wrapper=wrapper,
                ode_func=ode_func,
                sampler=sampler,
                noise_sampler=latent_sampler_eval,
                x0_batch=x_0,
                device=device,
                do_light=do_light,
                do_heavy=do_heavy,

            )

        ###################### EVAL IMAGES ######################
        if is_image_task:
            run_samples = (
                sample_vis_interval > 0
                and sample_vis_count > 0
                and ((step + 1) % sample_vis_interval == 0)
            )
            run_fid = (
                fid_interval > 0
                and fid_num_gen > 0
                and fid_real_cache is not None
                and ((step + 1) % fid_interval == 0)
            )
            if run_samples or run_fid:
                if fid_gen_batch > 0:
                    batch_size_for_logging = fid_gen_batch
                else:
                    fallback_bs = sample_vis_count if sample_vis_count > 0 else args.batch_size
                    batch_size_for_logging = max(1, fallback_bs)

                def generate_for_logging(
                    count: int,
                    *,
                    latents: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
                    return generate_baseline_samples(
                        count,
                        batch_size=batch_size_for_logging,
                        device=device,
                        dim=args.dim,
                        t_eval=t_eval,
                        ode_func=ode_func,
                        wrapper=wrapper,
                        eval_model=ema,
                        latent_sampler=latent_sampler_eval,
                        latents=latents,
                    )

                fixed_vis_noise = log_baseline_image_metrics(
                    args=args,
                    step=step,
                    eval_model=ema,
                    wrapper=wrapper,
                    device=device,
                    image_shape=image_shape,
                    sampler=sampler,
                    sample_vis_interval=sample_vis_interval,
                    sample_vis_count=sample_vis_count,
                    sample_vis_nrow=max(1, sample_vis_nrow),
                    sample_dir=sample_dir,
                    fid_interval=fid_interval,
                    fid_num_gen=fid_num_gen,
                    fid_batch_size=fid_batch_size,
                    fid_image_size=fid_image_size,
                    fid_gen_batch=fid_gen_batch,
                    fid_real_cache=fid_real_cache,
                    noise_sampler=latent_sampler_eval,
                    generate_samples=generate_for_logging,
                    fixed_noise=fixed_vis_noise,
                )
                if fixed_vis_noise is not None:
                    args._fixed_baseline_vis_noise = fixed_vis_noise
        
        current_step = step + 1
        if current_step % 20_000 == 0:
            ckpt_suffix = f"step_{current_step:06d}.pt"

            model_payload = {
                "step": current_step,
                "state_dict": model.state_dict(),
            }
            torch.save(model_payload, os.path.join(checkpoint_dir, f"model_{ckpt_suffix}"))

            if ema is not None:
                ema_payload = {
                    "step": current_step,
                    "state_dict": ema.state_dict(),
                }
                torch.save(ema_payload, os.path.join(checkpoint_dir, f"ema_{ckpt_suffix}"))



    final_step = args.epochs
    ckpt_suffix = f"step_{final_step:06d}.pt"
    model_payload = {
        "step": final_step,
        "state_dict": model.state_dict(),
    }
    torch.save(model_payload, os.path.join(checkpoint_dir, f"model_{ckpt_suffix}"))

    if ema is not None:
        ema_payload = {
            "step": final_step,
            "state_dict": ema.state_dict(),
        }
        torch.save(ema_payload, os.path.join(checkpoint_dir, f"ema_{ckpt_suffix}"))

    runtime_path = os.path.join(args.runs_dir, "runtime_training_only.txt")
    os.makedirs(args.runs_dir, exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as fh:
        fh.write(f"{train_time_accumulator:.6f}\n")
