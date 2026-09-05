# -*- coding: utf-8 -*-
import time
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker 
import pandas as pd
import random
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from scipy.stats import ks_2samp

from quantitative_comparison import compute_mmd

# =====================================================================
# 1. PIV Vorticity Plotting Tools
# =====================================================================
def plots_vort(U, vmin=-2, vmax=2):
    """Visualisiert ein 2D-Vorticity-Feld."""
    fig, axs = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    pcm = axs.pcolormesh(U[-1:0:-1, :], shading='auto', vmin=vmin, vmax=vmax)
    axs.set_title("vorticity (1/s)")
    axs.set_aspect('equal')
    fig.colorbar(pcm, ax=axs)


def plot_signal(xs, inds, prefix_save, std_norm, std_test_plot, plt_show=False, timeToDuplicate=None):
    """Plottet generierte Bilder oder Zeitserien über verschiedene Diffusionsschritte."""
    dim = xs[-1, :, :].shape[1]
    nb_samples = 10 if timeToDuplicate is not None else 1
    nb_samples = min((nb_samples, xs.shape[1]))
    if timeToDuplicate == -1:
        timeToDuplicate = xs.shape[0] - 1
        
    npixelx = np.int32(np.sqrt(dim))
    
    std_norm_np = std_norm.detach().cpu().numpy() if isinstance(std_norm, torch.Tensor) else np.asarray(std_norm)
    std_test_np = std_test_plot.detach().cpu().numpy() if isinstance(std_test_plot, torch.Tensor) else np.asarray(std_test_plot)
    factor_caxis = (std_norm_np * std_test_np).max()
    
    if dim > 4**2:
        if (dim == npixelx**2) and (npixelx >= 16):
            print("Plot noisy images")
            for ind in inds:
                nb_samples_loc = nb_samples if (ind == timeToDuplicate and timeToDuplicate is not None) else 1
                for id_sample in range(nb_samples_loc):
                    sample_vec = xs[ind, id_sample, :].squeeze().detach().cpu()
                    if isinstance(std_norm, torch.Tensor):
                        xtt_image = (std_norm.cpu() * sample_vec).numpy()
                    else:
                        xtt_image = std_norm * sample_vec.numpy()
                        
                    xtt_image = xtt_image.reshape(([npixelx, npixelx]), order='F')
                    plots_vort(xtt_image, -factor_caxis, factor_caxis)
                    if plt_show:
                        plt.show(block=False)
                    name_fig = f"{prefix_save}_imageAtt{ind}_sample{id_sample}_.png"
                    plt.savefig(name_fig)
                    if plt_show:
                        plt.pause(0.5)
                    plt.close('all')
        else:
            print("Plot noisy timeseries")
            time_axis = np.arange(0, dim)
            for ind in inds:
                nb_samples_loc = nb_samples if (ind == timeToDuplicate and timeToDuplicate is not None) else 1
                for id_sample in range(nb_samples_loc):
                    sample_vec = xs[ind, id_sample, :].squeeze().detach().cpu()
                    xtt_timeserie = (std_norm_np * sample_vec.numpy())
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(time_axis, xtt_timeserie)
                    ax.set_title(f"Noisy sample at step {ind}")
                    ax.set_xlabel("time")
                    ax.set_ylabel("Value")
                    ax.set_ylim(-2 * factor_caxis, 2 * factor_caxis)
                    plt.tight_layout()
                    if plt_show:
                        plt.show(block=False)
                    name_fig = f"{prefix_save}_timeserieAtt{ind}_sample{id_sample}_.png"
                    plt.savefig(name_fig)
                    if plt_show:
                        plt.pause(0.5)
                    plt.close('all')


# =====================================================================
# 2. Histogram & Pairplot Tools
# =====================================================================
@torch.no_grad()
def get_2d_histogram_plot(data, val=3, offset_dimplot=0, num=64, vmin=0, vmax=10, 
                          use_grid=False, origin='lower', logscale=True):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    x = data[:, offset_dimplot]
    if (offset_dimplot + data.shape[1]) < 3:
        y = data[:, offset_dimplot + 1]
    else:
        y = data[:, offset_dimplot + 2]
        val = val / 2

    xmin, xmax = -val, val
    ymin, ymax = -val, val

    heatmap, xedges, yedges = np.histogram2d(x, y, range=[[xmin, xmax], [ymin, ymax]], bins=num)
    if logscale:
        heatmap_val = heatmap.copy()
        if (heatmap > heatmap.min()).any():
            vmin = heatmap_val[heatmap > heatmap.min()].min() / 2
        heatmap = np.log(heatmap + 1e-10)
        vmin = np.log(vmin)
        vmax = heatmap.max()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(heatmap.T, extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    ax.grid(False)
    if use_grid:
        plt.xticks(np.arange(-val, val+1, step=1))
        plt.yticks(np.arange(-val, val+1, step=1))
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    tupl = fig.canvas.get_width_height()[::-1]
    if (tupl[0] * tupl[1] * 4 == image.shape[0]):
        image = image.reshape(tupl + (4,))
    else:
        image = image.reshape((tupl[0]*2, tupl[1]*2, 4))
    image = image[:, :, 1:]
    plt.close()
    return image


@torch.no_grad()
def plot_selected_inds(xs, inds, use_xticks=True, use_yticks=True, lmbd=0.,
                       offset_dimplot=0, include_t0=False, backward=True, plt_show=True, val=3):
    imgs_ = []
    l_inds = len(inds)
    if backward:
        inds = list(reversed(inds))
    for ind in inds:
        data_ind = xs[ind].detach().cpu().numpy() if isinstance(xs[ind], torch.Tensor) else xs[ind]
        val_scalar = val.item() if isinstance(val, torch.Tensor) else val
        imgs_ += [get_2d_histogram_plot(data_ind, val=val_scalar, offset_dimplot=offset_dimplot)]
    img_ = np.concatenate(imgs_, axis=1)

    height, width, _ = img_.shape
    height_per_img = width_per_img = height
    figwidth = 25
    fontsize = 15
    if use_xticks:
        xticks = [0.5*width_per_img + width_per_img*i for i in range(l_inds)]
        xticklabels = [rf'$i={ind+1:d}$' if not include_t0 else rf'$i={ind:d}$' for ind in inds]
    else:
        xticks, xticklabels = [], []
        
    yticks = [0.5*height_per_img] if use_yticks else []
    yticklabels = [rf'$\lambda={lmbd:.2g}$'] if use_yticks else []

    fig = plt.figure(figsize=(figwidth, figwidth*height/width))
    ax = fig.add_subplot(111)
    ax.imshow(img_)
    axis_color = 'white'
    for spine in ax.spines.values():
        spine.set_color(axis_color)
    ax.tick_params(axis='both', colors=axis_color)
    plt.xticks(xticks, xticklabels, color='black', fontsize=fontsize)
    plt.yticks(yticks, yticklabels, color='black', fontsize=fontsize)
    if plt_show:
        plt.show(block=False)


@torch.no_grad()
def def_pd(xgen, std_norm, std_test_plot, datatype, dimplot=2, offset_dimplot=0,
           crop_data_plot=False, plot_crop=3, columns_plot=None):
    device = xgen.device
    std_norm_t = torch.as_tensor(std_norm, device=device, dtype=xgen.dtype)
    std_test_plot_t = torch.as_tensor(std_test_plot, device=device, dtype=xgen.dtype)

    xgen_plot = std_norm_t * xgen
    if crop_data_plot:
        boolean_mask = (xgen_plot.abs() < (plot_crop * std_norm_t * std_test_plot_t)).all(dim=1)
        print(f"{(1 - boolean_mask.sum() / len(boolean_mask)).item() * 100} % of samples outside plot limits")
        xgen_plot = xgen_plot[boolean_mask, :]

    pddatagen = pd.DataFrame(xgen_plot[:, offset_dimplot:offset_dimplot+dimplot].detach().cpu().numpy(), columns=columns_plot)
    return pddatagen


@torch.no_grad()
def pairplots(xgen, xtest, std_norm, std_test_plot, datatype, name_simu, dimplot=2, offset_dimplot=0,
              crop_data_plot=False, plot_crop=3, plot_xlim=3, plot_ref_pdf=False,
              pdf_theor=None, log_scale_pdf=False, columns_plot=None,
              plt_show=False, dpi=200, height_seaborn=2.5, ssize=10):

    pddatatest = def_pd(xtest, std_norm, std_test_plot, datatype, dimplot=dimplot, offset_dimplot=offset_dimplot,
                        crop_data_plot=crop_data_plot, plot_crop=plot_crop, columns_plot=columns_plot)
    pddatagen = def_pd(xgen, std_norm, std_test_plot, datatype, dimplot=dimplot, offset_dimplot=offset_dimplot,
                       crop_data_plot=crop_data_plot, plot_crop=plot_crop, columns_plot=columns_plot)

    pddata = pd.concat([pddatatest.assign(samples="test"),
                        pddatagen.assign(samples="gen.")])

    palette = {"test": sns.color_palette()[0], "gen.": sns.color_palette()[1]}
    plot_kws = {'alpha': 0.1, "s": ssize, "edgecolor": "none", "rasterized": True}

    g = sns.PairGrid(pddata, hue="samples", corner=True, height=height_seaborn, aspect=1,
                     palette=palette, diag_sharey=False)
    g.map_lower(sns.scatterplot, **plot_kws)

    def to_scalar(val):
        if isinstance(val, torch.Tensor):
            return val.item()
        elif isinstance(val, (list, np.ndarray)):
            return val[0]
        return val

    def diag_plot(x, color=None, label=None, **kws):
        ax = plt.gca()
        if label == "test":
            x_np = np.asarray(x, dtype=np.float64)
            x_np = x_np[np.isfinite(x_np)]
            counts, _ = np.histogram(x_np, bins=80, density=True)
            ymax = float(counts.max()) if counts.size else 0.0

            sns.histplot(x=x, bins=80, stat="density", element="step", fill=True, alpha=0.25,
                         color=palette["test"], **kws)

            ymin = counts[counts > 0].min() if (log_scale_pdf and (counts > 0).any()) else 0
            if ymax > 0:
                ax.set_ylim(ymin, 1.05 * ymax)
        elif label == "gen.":
            sns.kdeplot(x=x, color=palette["gen."], lw=1.5, **kws)
        
        if log_scale_pdf:
            ax.set_yscale('log')

    g.map_diag(diag_plot)

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=palette[k], markersize=8, alpha=0.6) for k in ["test", "gen."]]
    g.figure.legend(handles=handles, labels=["test", "gen."], loc='upper right', markerscale=ssize)

    plt.tight_layout()
    if plt_show:
        plt.show(block=False); plt.pause(1)
    plt.savefig(name_simu + "_multDim.png", dpi=dpi)
    plt.close('all')


@torch.no_grad()
def pairplots_single(xtest, std_norm, std_test_plot, datatype, name_simu, dimplot=2, offset_dimplot=0,
                     crop_data_plot=False, plot_crop=3, plot_xlim=3, plot_ref_pdf=False,
                     pdf_theor=None, log_scale_pdf=False, columns_plot=None,
                     plt_show=False, dpi=200, height_seaborn=2.5, ssize=10):
    pddatatest = def_pd(xtest, std_norm, std_test_plot, datatype, dimplot=dimplot, offset_dimplot=offset_dimplot,
                        crop_data_plot=crop_data_plot, plot_crop=plot_crop, columns_plot=columns_plot)
    plot_kws = {"s": ssize}
    scatter = sns.pairplot(pddatatest, aspect=1, height=height_seaborn, corner=True, plot_kws=plot_kws)
    
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    if plt_show:
        plt.show(block=False); plt.pause(0.1)
    plt.savefig("results/" + name_simu + ".png", dpi=dpi)
    plt.close('all')


# =====================================================================
# 3. Survival Plotting Helpers
# =====================================================================
def _compute_common_R_grid(norms_list, n_points: int = 200) -> np.ndarray:
    mins, maxs = [], []
    for arr in norms_list:
        if arr is None or len(arr) == 0:
            continue
        pos = arr[arr > 0]
        if pos.size > 0:
            mins.append(pos.min())
            maxs.append(arr.max())
    if len(maxs) == 0:
        raise ValueError("No data provided to build R grid.")
    min_pos = min(mins) if len(mins) > 0 else 1e-12
    max_val = max(maxs)
    upper = max_val if max_val > min_pos else min_pos * 10.0
    return np.logspace(np.log10(min_pos * 0.9), np.log10(upper), num=n_points)


def _empirical_survival_from_norms(norms: np.ndarray, R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms_sorted = np.sort(norms)
    idx = np.searchsorted(norms_sorted, R_grid, side='right')
    counts = norms.size - idx
    S = counts.astype(float) / float(norms.size) if norms.size > 0 else np.zeros_like(R_grid)
    return S, counts


def _apply_std_norm(t: torch.Tensor, std_norm: Optional[torch.Tensor]) -> torch.Tensor:
    if std_norm is None:
        return t
    std_t = torch.as_tensor(std_norm, dtype=t.dtype, device=t.device) if not torch.is_tensor(std_norm) else std_norm.to(dtype=t.dtype, device=t.device)
    return t * std_t


def _tail_fit_loglog(R_grid: np.ndarray, S_vals: np.ndarray, norms: np.ndarray,
                     tail_frac: float = 0.05, tail_k: Optional[int] = None):
    n = norms.size
    if n < 10:
        return None, None, None
    sorted_norms = np.sort(norms)
    k = max(10, int(np.clip(np.ceil(n * tail_frac), 10, n - 1))) if tail_k is None else int(min(max(1, tail_k), n - 1))
    threshold = sorted_norms[-k - 1]
    mask = R_grid >= threshold
    if not np.any(mask):
        return None, k, None
    R_tail, S_tail = R_grid[mask], S_vals[mask]
    positive_mask = S_tail > 0
    if np.sum(positive_mask) < 3:
        return None, k, None
    b, a = np.polyfit(np.log(R_tail[positive_mask]), np.log(S_tail[positive_mask]), 1)
    return float(-b), int(k), np.exp(a) * (R_grid ** b)


def plot_survival_simple(x: Optional[torch.Tensor] = None, x_ref: Optional[torch.Tensor] = None,
                         std_norm: Optional[torch.Tensor] = None, prefix_save: str = "surv",
                         plt_show: bool = False, figsize: Tuple[float, float] = (3, 2),
                         n_points: int = 200, tail_frac: float = 0.05, tail_k: Optional[int] = None,
                         colors: Tuple[str, str] = ('#1f77b4', '#ff7f0e'),
                         ylim: Tuple[float, float] = (1e-3, 1.1), save_png: bool = True,
                         return_survival: bool = False, dpi: int = 300):
    sns.set_style("whitegrid")
    norms_gen, norms_ref = None, None
    if x_ref is not None:
        norms_ref = torch.norm(_apply_std_norm(x_ref, std_norm), dim=1).detach().cpu().numpy()
    if x is not None:
        norms_gen = torch.norm(_apply_std_norm(x, std_norm), dim=1).detach().cpu().numpy()

    R_grid = _compute_common_R_grid([norms_ref, norms_gen], n_points=n_points)
    S_ref, counts_ref = _empirical_survival_from_norms(norms_ref, R_grid) if norms_ref is not None else (None, None)
    S_gen, counts_gen = _empirical_survival_from_norms(norms_gen, R_grid) if norms_gen is not None else (None, None)

    fig, ax = plt.subplots(figsize=figsize)
    if S_ref is not None:
        ax.plot(R_grid, S_ref, linestyle='-', color=colors[0], label='test')
    if S_gen is not None:
        ax.plot(R_grid, S_gen, linestyle='-', color=colors[1], label='gen.')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('R')
    ax.set_ylabel(r"$S(R)=\mathbb{P}\left(\|\mathbf{x}\|>R\right)$")
    ax.set_ylim(max(ylim[0], 1e-300), ylim[1])
    ax.legend(frameon=False, loc='lower left', fontsize=7)
    plt.tight_layout()

    if save_png:
        fig.savefig(f"{prefix_save}_survival.png", bbox_inches='tight', dpi=dpi)
    if plt_show:
        plt.show(block=False)
    else:
        plt.close(fig)

    if return_survival:
        return fig, ax, {"R_grid": R_grid, "reference": {"S": S_ref}, "generated": {"S": S_gen}}
    return fig, ax


# =====================================================================
# 4. Preprocessing & Postprocessing Routinen
# =====================================================================
def preprocessing(xtest, xs_forward, num_steps_forward, name_simu_root,
                  noising_plots, plt_show, folder_results, val_hist, std_test_plot, device,
                  std_norm=None, offset_dimplot=0):
    xtest = xtest.to(device)
    xgen_forward = xs_forward[-1, :, :].to(device)

    cov_xtest = torch.cov(xtest.T)
    cov_xgen_forward = torch.cov(xgen_forward.T)
    xgen_forward_var_mean = torch.var(xgen_forward.T, dim=1).mean()
    xtest_var_mean = torch.var(xtest.T, dim=1).mean()

    cov_xgen_forward_converged = xtest_var_mean * torch.eye(xtest.shape[1], device=device, dtype=xtest.dtype)
    d_cov_xtest = torch.norm(cov_xtest - cov_xgen_forward_converged) / torch.norm(cov_xgen_forward_converged)
    d_cov_xgen_forward = torch.norm(cov_xgen_forward - cov_xgen_forward_converged) / torch.norm(cov_xgen_forward_converged)
    
    print(f"dist cov_xtest to cov_xgen_forward_converged = {d_cov_xtest.item():.6f}")
    print(f"dist cov_xgen_forward to cov_xgen_forward_converged = {d_cov_xgen_forward.item():.6f}")

    fig_step = max(1, int(num_steps_forward / 8))
    inds_forward = range(0, num_steps_forward + 1, fig_step)
    
    if noising_plots:
        os.makedirs(folder_results, exist_ok=True)
        val_scalar = (val_hist * std_test_plot[0])
        val_scalar = val_scalar.item() if isinstance(val_scalar, torch.Tensor) else val_scalar
            
        plot_selected_inds(xs_forward, inds_forward, use_xticks=True, use_yticks=False, lmbd=0.,
                           include_t0=True, backward=False, plt_show=plt_show, val=val_scalar)
        plt.savefig(f"{folder_results}/{name_simu_root}_Forward.png")
        plt.close('all')

        if std_norm is not None:
            prefix_save = f"{folder_results}/{name_simu_root}_Forward"
            plot_signal(xs_forward, inds_forward, prefix_save, std_norm=std_norm,
                        std_test_plot=std_test_plot, plt_show=plt_show, timeToDuplicate=0)

def postprocessing(inds, i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run, MSGM, sampler,
                   xs, xtest, std_norm, std_test_plot, datatype, name_simu, dimplot,
                   crop_data_plot, plot_crop, plot_xlim, plot_ref_pdf,
                   pdf_theor, log_scale_pdf, columns_plot,
                   scatter_plots, denoising_plots, include_t0_reverse, plt_show, dpi, height_seaborn, ssize,
                   evalmmmd, justLoadmmmd, justLoad, save_results, lmbd, val_hist, device,
                   mmd_ref, mmd_MSGM, mmd_SGM, rad_w1_MSGM, sliced_w1_MSGM, ks_MSGM, max_num_samples_for_mmd, offset_dimplot=0, model_name=""):

    xtest = xtest.to(device)
    xgen = xs[-1, :, :].to(device)

    import os
    
    # --- DYNAMISCHE DIMENSION BERECHNEN ---
    # Nimmt an, dass die Daten flache Vektoren sind, deren Länge ein perfektes Quadrat ist.
    vec_len = xtest.shape[1]
    grid_size = int(np.sqrt(vec_len))
    if grid_size * grid_size != vec_len:
        print(f"WARNUNG: Daten-Dimension {vec_len} ist kein perfektes Quadrat. Reshape wird fehlschlagen!")

    # =================================================================
    # DIAGNOSE: Wie viele der 1000 Samples sind eigentlich kaputt?
    # =================================================================
    print("\n--- STARTE BATCH-DIAGNOSE ---")
    xgen_np = xgen.detach().cpu().numpy()
    
    smoothness_list = []
    for j in range(xgen_np.shape[0]):
        img = xgen_np[j].reshape((grid_size, grid_size), order='F')
        diff_y = np.mean(np.abs(img[1:, :] - img[:-1, :]))
        diff_x = np.mean(np.abs(img[:, 1:] - img[:, :-1]))
        smoothness_list.append((diff_x + diff_y) / 2.0)
        
    smoothness_arr = np.array(smoothness_list)
    
    # Wir wissen: Gute PIV-Bilder haben eine Smoothness von ca. 0.1 bis 0.25
    # Rauschen hat eine Smoothness von > 0.5
    good_mask = smoothness_arr < 0.35
    
    num_good = good_mask.sum()
    num_bad = (~good_mask).sum()
    
    print(f"Total Samples im Batch: {len(smoothness_arr)}")
    print(f"GUTE Samples (Smoothness < 0.35): {num_good}")
    print(f"KAPUTTE Samples (Smoothness >= 0.35): {num_bad}")
    
    if num_bad > 0:
        bad_indices = np.where(~good_mask)[0]
        print(f"Die ersten 10 kaputten Indizes: {bad_indices[:10]}")
    print("-----------------------------\n")
    
    print("\n" + "*" * 70)
    print(f"=== REDUCED SANITY CHECK: 1 Sample ({grid_size}x{grid_size}), Order F only ===")
    
    debug_dir = os.path.abspath("DEBUG_SanityCheck")
    os.makedirs(debug_dir, exist_ok=True)

    # Hilfsfunktion für die Smoothness (Total Variation)
    def calc_smoothness(img_2d):
        # Mittlere absolute Differenz zu direkten Nachbarn (x und y Richtung)
        diff_y = np.mean(np.abs(img_2d[1:, :] - img_2d[:-1, :]))
        diff_x = np.mean(np.abs(img_2d[:, 1:] - img_2d[:, :-1]))
        return (diff_x + diff_y) / 2.0

    # Nur noch 1 Sample prüfen (spart Zeit und Speicher)
    if xtest.shape[0] > 0 and xs.shape[1] > 0:
        i = 0
        print(f"--- Auswertung für Sample {i} ---")
        
        # --- Echte Testdaten (xtest) ---
        test_vec = xtest[i].detach().cpu().numpy()
        test_img_f = test_vec.reshape((grid_size, grid_size), order='F')
        
        plots_vort(test_img_f, vmin=-2, vmax=2)
        plt.savefig(os.path.join(debug_dir, f"DEBUG_Test_Sample{i}_OrderF.png"))
        plt.close('all')

        # --- Generierte Daten am ENDE der SDE (xs[-1]) ---
        gen_end_vec = xs[-1, i, :].detach().cpu().numpy()
        gen_end_img_f = gen_end_vec.reshape((grid_size, grid_size), order='F')
        
        plots_vort(gen_end_img_f, vmin=-2, vmax=2)
        plt.savefig(os.path.join(debug_dir, f"DEBUG_Gen_Step-1_Sample{i}_OrderF.png"))
        plt.close('all')
        
        # --- Statistiken und Smoothness vergleichen ---
        smooth_test = calc_smoothness(test_img_f)
        smooth_end = calc_smoothness(gen_end_img_f)

        print(f"  TEST Bild     -> Mean: {test_vec.mean():.4f} | Smoothness (Gradient): {smooth_test:.4f}")
        print(f"  GEN Bild [-1] -> Mean: {gen_end_vec.mean():.4f} | Smoothness (Gradient): {smooth_end:.4f}")
        
    print("*" * 70 + "\n")

    if save_results and not justLoad:
        np.save(name_simu + ".npy", xgen.clone().detach().cpu().numpy())

    nan_mask = (torch.isnan(xgen) | (torch.abs(xgen) > 1e3)).any(dim=1)
    nan_count = nan_mask.sum().item()
    if nan_count > 0:
        print(f"Number of rows with NaN or large value: {nan_count}")
    xgen = xgen[~nan_mask, :]
    del nan_mask

    if scatter_plots and (i_run == 0):
        pairplots(xgen, xtest, std_norm, std_test_plot, datatype, name_simu, dimplot=dimplot, offset_dimplot=offset_dimplot,
                  crop_data_plot=crop_data_plot, plot_crop=plot_crop, plot_xlim=plot_xlim, plot_ref_pdf=plot_ref_pdf,
                  pdf_theor=pdf_theor, log_scale_pdf=log_scale_pdf, columns_plot=columns_plot,
                  plt_show=plt_show, dpi=dpi, height_seaborn=height_seaborn, ssize=ssize)

    # Survival function plot
    plot_survival_simple(x=xgen, x_ref=xtest, std_norm=std_norm, prefix_save=name_simu, plt_show=False)

    if denoising_plots and (i_run == 0):
        val_scalar = (val_hist * std_test_plot[0])
        val_scalar = val_scalar.item() if isinstance(val_scalar, torch.Tensor) else val_scalar
            
        plot_selected_inds(xs, inds, True, False, lmbd, include_t0=include_t0_reverse,
                           plt_show=plt_show, val=val_scalar, offset_dimplot=offset_dimplot)
        plt.savefig(name_simu + ".png")
        plt.close('all')
    
    # Generierte Bilder visualisieren
    prefix_save = name_simu + "_Gen"
    plot_signal(xs, inds, prefix_save, std_norm=std_norm, std_test_plot=std_test_plot,
                plt_show=plt_show, timeToDuplicate=-1)

    # =================================================================
    # MMD Evaluation (10 Seeds & 64-Bit Präzision)
    # =================================================================
    if evalmmmd and not justLoadmmmd:
        num_samples_for_mmd = min([xtest.shape[0], max_num_samples_for_mmd])
        print(f"Evaluating Metrics with {num_samples_for_mmd} samples across 10 seeds")
        
        xgen_sub = xgen[0:num_samples_for_mmd-1, :]
        std_norm_t = torch.as_tensor(std_norm, device=device, dtype=xtest.dtype)

        # Liste von 10 Seeds (966 bleibt als Anker dabei)
        evaluation_seeds = [966, 41, 122, 776, 1023, 2022, 3140, 8081, 9998, 12344, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results_records = []

        for current_seed in evaluation_seeds:
            with torch.no_grad():
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                random.seed(current_seed)
                
                # Ziehe Test- und Train-Samples mit dem aktuellen Seed
                xtest_baseline = sampler.sampletest(2500).to(device)
                xtest_baseline_sub = xtest_baseline[0:num_samples_for_mmd-1, :]
                x_mmd1 = sampler.sample(xtest_baseline_sub.shape[0]).to(device)
                
                # Denormierte Versionen
                x_gen_eval = std_norm_t * xgen_sub
                x_test_eval = std_norm_t * xtest_baseline_sub
                
                # --- 1. MMD ---
                dist_train_to_test = compute_mmd(std_norm_t * x_mmd1, x_test_eval)
                dist_gen_to_test = compute_mmd(x_gen_eval, x_test_eval)

                # --- 2. Radial Wasserstein-1 ---
                r_gen = torch.linalg.vector_norm(x_gen_eval, dim=1)
                r_test = torch.linalg.vector_norm(x_test_eval, dim=1)
                r_train = torch.linalg.vector_norm(std_norm_t * x_mmd1, dim=1)
                
                r_gen_sorted = torch.sort(r_gen)[0]
                r_test_sorted = torch.sort(r_test)[0]
                r_train_sorted = torch.sort(r_train)[0]
                
                radial_w1 = torch.abs(r_gen_sorted - r_test_sorted).mean().item()
                radial_w1_baseline = torch.abs(r_train_sorted - r_test_sorted).mean().item()
                
                # --- 3. KS Statistic ---
                ks_stat, _ = ks_2samp(r_gen.cpu().numpy(), r_test.cpu().numpy())
                ks_stat_baseline, _ = ks_2samp(r_train.cpu().numpy(), r_test.cpu().numpy())
                
                # --- 4. Sliced Wasserstein-1 ---
                num_projections = 500
                dim = x_gen_eval.shape[1]
                
                directions = torch.randn(dim, num_projections, device=device)
                directions = torch.nn.functional.normalize(directions, dim=0)
                
                proj_gen = torch.matmul(x_gen_eval, directions)
                proj_test = torch.matmul(x_test_eval, directions)
                proj_train = torch.matmul(std_norm_t * x_mmd1, directions)
                
                proj_gen_sorted = torch.sort(proj_gen, dim=0)[0]
                proj_test_sorted = torch.sort(proj_test, dim=0)[0]
                proj_train_sorted = torch.sort(proj_train, dim=0)[0]
                
                sliced_w1 = torch.abs(proj_gen_sorted - proj_test_sorted).mean().item()
                sliced_w1_baseline = torch.abs(proj_train_sorted - proj_test_sorted).mean().item()

                # Speichere die Ergebnisse dieses Durchlaufs
                results_records.append({
                    "Seed": current_seed,
                    "MMD_Base": dist_train_to_test**0.5,
                    "MMD_Mod": dist_gen_to_test**0.5,
                    "RadW1_Base": radial_w1_baseline,
                    "RadW1_Mod": radial_w1,
                    "SW1_Base": sliced_w1_baseline,
                    "SW1_Mod": sliced_w1,
                    "KS_Base": ks_stat_baseline,
                    "KS_Mod": ks_stat,
                    # Raw Items (ohne sqrt) für die globalen Arrays
                    "raw_mmd_base": dist_train_to_test.item(),
                    "raw_mmd_mod": dist_gen_to_test.item()
                })

        # --- 5. Tabelle ausgeben ---
        df_results = pd.DataFrame(results_records)
        
        print("\n" + "="*95)
        print(f"EVALUATION RESULTS ACROSS {len(evaluation_seeds)} SEEDS")
        print("="*95)
        # Die rohen MMD-Werte blenden wir für die hübsche Print-Tabelle aus
        print(df_results.drop(columns=["raw_mmd_base", "raw_mmd_mod"]).to_string(index=False, float_format="{:.5f}".format))
        print("="*95)
        
        # --- 6. Durchschnittswerte und Std-Abweichung berechnen und ausgeben ---
        mean_results = df_results.mean().to_dict()
        std_results = df_results.std().to_dict()
        
        print("\nAVERAGE OVER ALL SEEDS (Mean +/- Std):")
        for k in mean_results.keys():
            if k not in ["Seed", "raw_mmd_base", "raw_mmd_mod"]:
                m_val = mean_results[k]
                s_val = std_results[k]
                print(f"{k:15s}: {m_val:.6f} +/- {s_val:.6f}")
        print("="*95 + "\n")

        # --- 7. Werte in die globalen Tensoren schreiben (Mittelwerte) ---
        # Wir speichern die gemittelten Werte, damit die restlichen Plots funktionieren
        
        # MMD speichern (Hier wird der rohe Durchschnitt gespeichert, nicht der sqrt-Durchschnitt)
        mmd_ref[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["raw_mmd_base"]


        # Modell-Werte speichern (aufgeteilt nach SGM / MSGM)
        if MSGM:
            mmd_MSGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["raw_mmd_mod"]
            rad_w1_MSGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["RadW1_Mod"]
            sliced_w1_MSGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["SW1_Mod"]
            ks_MSGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = float(mean_results["KS_Mod"])
        else:
            mmd_SGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["raw_mmd_mod"]
            # ACHTUNG: Hier ggf. rad_w1_SGM etc. ergänzen, falls du sie oben initialisiert hast!
            # rad_w1_SGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["RadW1_Mod"]
            # sliced_w1_SGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = mean_results["SW1_Mod"]
            # ks_SGM[i_dims, i_Res, i_num_stepss_backward, i_iterations, i_run] = float(mean_results["KS_Mod"])

    # =================================================================
    # ANIMATION: 4 GIFs erzeugen (jedes 4. Frame + Start & Ende)
    # =================================================================
    import shutil

    print("\n=== Erstelle 4 GIF-Animationen (reduzierte Framerate) ===")
    
    gif_dir = os.path.abspath("SHOWCASE_GIFs")
    os.makedirs(gif_dir, exist_ok=True)
    
    # Bereinigter Name für die Datei (ohne Pfad-Schrägstriche)
    clean_name = os.path.basename(name_simu)
    
    # Wie viele GIFs wollen wir maximal? (4 Stück)
    num_gifs = min(2, xs.shape[1]) 
    
    for sample_idx in range(num_gifs):
        print(f"\nGeneriere GIF für Sample {sample_idx+1}/{num_gifs}...")
        
        trajectory = xs[:, sample_idx, :].detach().cpu().numpy()
        num_steps = trajectory.shape[0]
        
        # Logik: Jedes 4. Bild (0, 4, 8, 12...)
        frame_indices = list(range(0, num_steps, 4))
        
        # WICHTIG: Sicherstellen, dass das allerletzte Bild (t=0) auf jeden Fall dabei ist!
        if frame_indices[-1] != num_steps - 1:
            frame_indices.append(num_steps - 1)
            
        frames = []
        temp_dir = os.path.join(gif_dir, f"temp_frames_{sample_idx}")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, step in enumerate(frame_indices):
            # Zeitschritt in dynamisches Bild umwandeln
            img_f = trajectory[step].reshape((grid_size, grid_size), order='F')
            
            plots_vort(img_f, vmin=-2, vmax=2)
            
            # Titel mit Sample-Nummer und aktuellem Step
            plt.title(f"Sample {sample_idx+1} | Step {step}/{num_steps-1}", fontsize=14, color='white')
            
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path, bbox_inches='tight', dpi=100)
            plt.close('all')
            
            frames.append(Image.open(frame_path))
            
        gif_path = os.path.join(gif_dir, f"sample{sample_idx+1}_{model_name}.gif")
        
        # Dauer anpassen: Da wir nur 1/4 der Frames haben, zeigen wir jedes Frame etwas länger (z.B. 150ms), 
        # damit die Animation insgesamt gut sichtbar bleibt und nicht in 1 Sekunde vorbei ist.
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=150, 
            loop=0
        )
        
        # Räume den Ordner mit den Einzelbildern für dieses Sample sofort wieder auf
        shutil.rmtree(temp_dir)
        print(f"-> Gespeichert: {gif_path}")
        
    print("\nAlle 2 GIFs wurden erfolgreich erstellt und die Temp-Ordner aufgeräumt!\n")