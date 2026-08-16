from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable
import time

import numpy as np
import torch
import wandb

from learn_noise.configs import apply_overrides, list_configs, load_config, parse_override_strings
from learn_noise.networks import UNetModel, VelocityFieldAdapter
from learn_noise.training import pretrain_quantile, train_fm_baseline, train_fm_quantile
from learn_noise.networks.model_wrapper import SphericalProjectedModel, NormalizedRadiusConditionedModel, MinimalRadiusConcatWrapper, LogFourierMLPWrapper


def _build_unet(args: SimpleNamespace) -> torch.nn.Module:
    image_shape = tuple(args.image_shape)
    num_classes = None
    if getattr(args, "class_conditional", False):
        num_classes = int(getattr(args, "num_classes", 0)) or None

    dataset = args.target_dataset.lower()
    
    # === HIER HABE ICH msgm_piv HINZUGEFÜGT ===
    if dataset in {"cifar10", "cifar", "cifar-10", "msgm_piv"}:
        model_channels = int(getattr(args, "unet_model_channels", 64)) # Für 1-Kanal PIV auf 64 (statt 128) reduziert, damit es schneller trainiert
        channel_mult = tuple(getattr(args, "unet_channel_mult", (1, 2, 2, 2)))
        num_res_blocks = int(getattr(args, "unet_num_res_blocks", 2))
        attention_resolutions = tuple(getattr(args, "unet_attention_resolutions", (16,)))
        num_heads = int(getattr(args, "unet_num_heads", 4))
        num_head_channels = int(getattr(args, "unet_num_head_channels", 64))
        dropout = float(getattr(args, "unet_dropout", 0.1))
        
        # Flexibler Input-Kanal (holt die '1' aus image_shape [1, 32, 32])
        in_channels = int(getattr(args, "unet_in_channels", image_shape[0]))
        out_channels = int(getattr(args, "unet_out_channels", in_channels))
        
        base_model = UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_shape[-1],
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            num_classes=num_classes,
        )
    elif dataset == "mnist":
        model_channels = int(getattr(args, "unet_model_channels", 64))
        # ... (MNIST Setup bleibt unverändert) ...
        channel_mult = tuple(getattr(args, "unet_channel_mult", (1, 2, 4)))
        num_res_blocks = int(getattr(args, "unet_num_res_blocks", 2))
        attention_resolutions = tuple(getattr(args, "unet_attention_resolutions", (7,)))
        num_heads = int(getattr(args, "unet_num_heads", 1))
        num_head_channels = int(getattr(args, "unet_num_head_channels", 16))
        dropout = float(getattr(args, "unet_dropout", 0.0))
        in_channels = int(getattr(args, "unet_in_channels", 1))
        out_channels = int(getattr(args, "unet_out_channels", in_channels))
        base_model = UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_shape[-1],
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported image dataset: {args.target_dataset}")

    device = torch.device(args.device)
    base_model = base_model.to(device)
    # VelocityFieldAdapter wandelt (wahrscheinlich) 1024D Vektoren ins 1x32x32 Bildformat und zurück!
    return VelocityFieldAdapter(base_model, image_shape).to(device)


def _make_namespace(cfg: Dict[str, object]) -> SimpleNamespace:
    cfg = dict(cfg)
    image_shape = cfg.get("image_shape")
    if image_shape in (None, "null"):
        cfg["image_shape"] = None
    else:
        cfg["image_shape"] = tuple(image_shape)
        
    if "unet_channel_mult" in cfg and cfg["unet_channel_mult"] is not None:
        cfg["unet_channel_mult"] = tuple(cfg["unet_channel_mult"])
    if "unet_attention_resolutions" in cfg and cfg["unet_attention_resolutions"] is not None:
        cfg["unet_attention_resolutions"] = tuple(cfg["unet_attention_resolutions"])
        
    cfg.setdefault("latent_viz_samples", 0)
    cfg.setdefault("latent_atlas_grid", 1)
    cfg.setdefault("sample_vis_interval", 0)
    cfg.setdefault("sample_vis_count", 0)
    cfg.setdefault("sample_vis_nrow", 8)
    cfg.setdefault("fid_eval_interval", 0)
    cfg.setdefault("fid_num_gen", 0)
    cfg.setdefault("fid_batch_size", cfg.get("batch_size", 0))
    cfg.setdefault("fid_gen_batch", cfg.get("batch_size", 0))
    
    default_image_size = cfg["image_shape"][-1] if cfg.get("image_shape") else 0
    cfg.setdefault("fid_image_size", default_image_size)
    cfg.setdefault("lambda_reg", 0.0)
    
    cfg.setdefault("student_t_df", 4.0)
    cfg.setdefault("student_t_scale", 1.0)
    cfg.setdefault("quantile_const_iters", None)
    cfg.setdefault("quantile_decay_iters", None)
    return SimpleNamespace(**cfg)


def _init_wandb(args: SimpleNamespace, run_name: str, disabled: bool) -> None:
    if disabled:
        wandb.init(mode="disabled")
        return
    project = getattr(args, "wandb_project", None)

    if not project:
        wandb.init(mode="disabled")
        return
    wandb_kwargs = {
        "project": project,
        "entity": getattr(args, "wandb_entity", None),
        "group": getattr(args, "wandb_group", None),
        "mode": "online",
        "name": run_name,
        "config": vars(args),
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)


def _save_config(run_dir: Path, cfg: Dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    import yaml

    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn-Noise Image & 2D MSGM experiments")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--pretrain", action="store_true", help="Run quantile pretraining before joint training")
    parser.add_argument("--baseline", action="store_true", help="Run baseline FM instead of quantile FM")
    parser.add_argument("--dataset", type=str, help="Override target_dataset in the config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--quantile-checkpoint", type=str)
    parser.add_argument("--freeze-quantile", action="store_true")
    parser.add_argument("--use-minibatch-ot", action="store_true")
    
    # === SLERP / MSGM Args aus deiner 2D Main ===
    parser.add_argument("--slerp", action="store_true")
    parser.add_argument("--strat_ot", action="store_true")
    parser.add_argument("--msgm", action="store_true", help="Train the MSGM Baseline instead of FM")
    parser.add_argument("--msgm-beta-min", type=float, default=0.1)
    parser.add_argument("--msgm-beta-max", type=float, default=2.0)
    parser.add_argument("--msgm-t-eps", type=float, default=1e-3)
    parser.add_argument("--msgm-steps", type=int, default=16)
    # =============================================
    
    parser.add_argument("--baseline-flow", type=str, choices=["linear", "mmd", "kac"])
    parser.add_argument("--baseline-latent", type=str)
    parser.add_argument("--baseline-flow-T", type=float)
    parser.add_argument("--baseline-mmd-b", type=float)
    parser.add_argument("--baseline-kac-a", type=float)
    parser.add_argument("--baseline-kac-c", type=float)
    parser.add_argument("--baseline-kac-epsilon", type=float)
    parser.add_argument("--baseline-kac-lookup-M", type=int)
    parser.add_argument("--baseline-kac-lookup-K", type=int)
    parser.add_argument("--q-loss-weight", type=float, help="Override q_loss_weight")
    parser.add_argument(
        "--q-objective",
        type=str,
        choices=["energy", "sinkhorn", "plan_action"],
        help="Override quantile OT objective during pretraining",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries (top-level only)",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    # Wir nutzen hier list_configs aus "images", da du U-Nets baust
    try:
        base_cfg = load_config("images", cli_args.config)
    except:
        # Fallback falls deine config in einem "2d" Ordner liegt
        base_cfg = load_config("2d", cli_args.config)
        
    overrides = parse_override_strings(cli_args.override)

    if cli_args.dataset is not None:
        overrides["target_dataset"] = cli_args.dataset
    if cli_args.seed is not None:
        overrides["seed"] = cli_args.seed
    if cli_args.device is not None:
        overrides["device"] = cli_args.device
    if cli_args.quantile_checkpoint is not None:
        overrides["quantile_checkpoint"] = cli_args.quantile_checkpoint
    if cli_args.freeze_quantile:
        overrides["freeze_quantile"] = True
    if cli_args.use_minibatch_ot:
        overrides["use_minibatch_ot"] = True
    if cli_args.slerp:
        overrides["slerp"] = True
    if cli_args.strat_ot:
        overrides["strat_ot"] = True
    if cli_args.baseline_flow is not None:
        overrides["baseline_flow"] = cli_args.baseline_flow
    if cli_args.baseline_latent is not None:
        overrides["baseline_latent"] = cli_args.baseline_latent
    if cli_args.baseline_flow_T is not None:
        overrides["baseline_flow_T"] = cli_args.baseline_flow_T
    if cli_args.wandb_project is not None:
        overrides["wandb_project"] = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        overrides["wandb_entity"] = cli_args.wandb_entity
    if cli_args.name is not None:
        overrides["name"] = cli_args.name

    apply_overrides(base_cfg, overrides)

    dataset = base_cfg["target_dataset"]
    run_suffix = base_cfg.get("name") or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset}-{run_suffix}"

    run_dir = cli_args.output_root / dataset / run_name
    base_cfg["runs_dir"] = str(run_dir)

    # Namespace for trainers
    args = _make_namespace(base_cfg)
    args.is_msgm = getattr(cli_args, "msgm", False)
    args.mode = "fm_baseline" if cli_args.baseline else "fm_and_quantile"

    if args.is_msgm:
        args.msgm_beta_min = cli_args.msgm_beta_min
        args.msgm_beta_max = cli_args.msgm_beta_max
        args.msgm_t_eps = cli_args.msgm_t_eps
        args.msgm_steps = cli_args.msgm_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _save_config(run_dir, base_cfg)

    wandb_disabled = cli_args.no_wandb or not bool(base_cfg.get("log_wandb", True))
    if not wandb_disabled:
        print("Initializing Weights & Biases...")
    _init_wandb(args, run_name, wandb_disabled)
    
    args.input_dim = args.dim

    if not args.is_msgm:
        print(f"Building U-Net for {dataset}...")
        model = _build_unet(args)
        
        # === DEIN SLERP WRAPPER FÜR DAS U-NET ===
        if getattr(args, "slerp", False):
            print("Applying SLERP (SphericalProjectedModel) wrapper...")
            args.baseline_flow = "target_norm_emp"
            model = SphericalProjectedModel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = None 
        optimizer = None
        args.slerp = False
        args.strat_ot = False
        args.baseline_flow = "target_norm_emp"

    if cli_args.pretrain:
        pretrain_cfg = dict(base_cfg)
        pretrain_cfg["runs_dir"] = str(run_dir / "pretrain")
        pretrain_args = _make_namespace(pretrain_cfg)
        pretrain_args.mode = "pretrain_quantile"
        _save_config(Path(pretrain_args.runs_dir), pretrain_cfg)
        _, _, pretrain_steps = pretrain_quantile(pretrain_args)
        args._pretrain_step_offset = int(getattr(pretrain_args, "_pretrain_step_offset", pretrain_steps))
        quant_ckpt = Path(pretrain_args.runs_dir) / "quantile_ot" / "quantile.pt"
        if quant_ckpt.exists():
            args.quantile_checkpoint = str(quant_ckpt)

    args.runs_dir = str(run_dir)
    print(f"Seed is {args.seed}")
    
    if cli_args.baseline:
        train_fm_baseline(args, model, optimizer)
    else:
        train_fm_quantile(args, model, optimizer)

    if not wandb_disabled:
        wandb.finish()


if __name__ == "__main__":
    main()