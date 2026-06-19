"""
CLI + YAML configuration for the density-estimation training pipeline.

Resolution order (highest priority first):
    1. CLI arguments          (``--lr 5e-4``)
    2. YAML config file       (``--config path/to/config.yaml``)
    3. Hardcoded defaults     (defined below)

This mirrors the pattern used in ``src.utils.config_parser`` for the
segmentation pipeline.
"""

from __future__ import annotations

import argparse
import os
import pprint
from typing import Any, Dict

import yaml
from loguru import logger

# ============================================================
# Hardcoded defaults — used when neither CLI nor YAML provides
# a value.  Grouped by section matching the YAML layout.
# ============================================================

# --- IO ---
DEFAULT_ROOT_DIR: str | None = None  # required
DEFAULT_OUTPUT_DIR: str | None = None  # required

# --- Training ---
DEFAULT_NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_SEED = 42

# --- Loss ---
DEFAULT_LOSS_CONFIGS = [{"type": "mse", "weight": 1.0}]

# --- Optimizer ---
DEFAULT_OPTIMIZER_TYPE = "adam"
DEFAULT_OPTIMIZER_KWARGS: Dict[str, Any] = {}

# --- Gradient Clipping ---
DEFAULT_GRAD_CLIP_MAX_NORM: float | None = None  # None = disabled

# --- Scheduler ---
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_SCHEDULER_MIN_LR = 1e-7

# --- Cross Validation ---
DEFAULT_K_FOLDS = 5

# --- Early Stopping ---
DEFAULT_EARLY_STOPPING_PATIENCE = 30
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001
DEFAULT_EARLY_STOPPING_MODE = "min"  # 'min', 'max', or 'auto' (inferred from metric)
DEFAULT_EARLY_STOPPING_MONITOR = "val_nae"  # metric key to monitor

# --- Model ---
DEFAULT_MODEL_TYPE = "plain_conv_unet"
DEFAULT_IMG_SIZE = 128
DEFAULT_NUM_CLASSES = 3
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_DEEP_SUPERVISION = False

# Default UNet architecture kwargs (PlainConvUNet)
DEFAULT_MODEL_KWARGS: Dict[str, Any] = {
    "n_stages": 4,
    "features_per_stage": [16, 32, 64, 128],
    "strides": [1, 2, 2, 2],
    "n_conv_per_stage": [2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2],
    "kernel_sizes": [3, 3, 3, 3],
    "conv_bias": True,
    "norm_op": "InstanceNorm2d",
    "nonlin": "LeakyReLU",
    "use_log_counts": False,  # When True, targets are log1p-compressed; default raw counts
}

# --- Augmentations ---
DEFAULT_AUG_LEVEL = "basic"

# --- Data ---
DEFAULT_CLASS_NAMES = ["Pyramidal", "Interneuron", "Astrocyte"]
DEFAULT_NORM_MEAN = [0.7637, 0.7637, 0.7637]
DEFAULT_NORM_STD = [0.0703, 0.0703, 0.0703]
DEFAULT_FILL_VALUE = 65535
DEFAULT_USE_ROI_MASK = False  # Focus loss/metrics on ROI regions only

# --- WandB ---
DEFAULT_USE_WANDB = False
DEFAULT_WANDB_PROJECT = "neuro_brain_project"
DEFAULT_WANDB_GROUP = "density_estimator"
DEFAULT_WANDB_TAGS: list[str] = []
DEFAULT_WANDB_RUN_NAME: str | None = None  # auto-generated from YAML name + timestamp

# --- Debug ---
DEFAULT_DEBUG = False


# ============================================================
# Helper
# ============================================================


def _str2bool(v: Any) -> bool:
    """Convert flexible boolean strings to ``bool``."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _validate_channel_selection(
    parser: argparse.ArgumentParser,
    num_classes: int,
    channel_to_predict: int | None,
    class_names: list[str],
) -> None:
    """Validate single-channel selection against the available class channels."""
    if num_classes != 1:
        return

    if channel_to_predict is None:
        parser.error("--channel-to-predict is required when --num-classes=1.")

    if not class_names:
        parser.error(
            "--class-names must define the available class channels when --num-classes=1."
        )

    if not 0 <= channel_to_predict < len(class_names):
        parser.error(
            "--channel-to-predict must be between 0 and "
            f"{len(class_names) - 1} for the configured class channels {class_names}."
        )


# ============================================================
# Main parser
# ============================================================


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Build the argument namespace from YAML + CLI.

    Priority: CLI flags  >  YAML values  >  hardcoded defaults.

    Args:
        argv: Explicit argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Fully-resolved ``argparse.Namespace``.
    """
    # ---------------------------------------------------------
    # 1. Pre-parse: extract --config path (optional)
    # ---------------------------------------------------------
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    known_args, remaining_argv = conf_parser.parse_known_args(argv)

    # ---------------------------------------------------------
    # 2. Load YAML sections (with safe fallbacks)
    # ---------------------------------------------------------
    config: Dict[str, Any] = {}
    if known_args.config and os.path.exists(known_args.config):
        logger.info(f"Loading configuration from {known_args.config}")
        with open(known_args.config, "r") as fh:
            config = yaml.safe_load(fh) or {}
    elif known_args.config:
        logger.warning(f"Config file {known_args.config} not found — using defaults.")

    io_c = config.get("IO", {})
    train_c = config.get("TRAINING", {})
    loss_c = config.get("LOSS", {})
    optim_c = config.get("OPTIMIZER", {})
    sched_c = config.get("SCHEDULER", {})
    cv_c = config.get("CROSS_VALIDATION", {})
    es_c = config.get("EARLY_STOPPING", {})
    model_c = config.get("MODEL", {})
    data_c = config.get("DATA", {})
    aug_c = config.get("AUGMENTATIONS", {})
    wandb_c = config.get("WANDB", {})

    # ---------------------------------------------------------
    # 3. Main parser — defaults from YAML, then fallback
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Density-estimation training pipeline.",
        parents=[conf_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    io_g = parser.add_argument_group("I/O")
    io_g.add_argument(
        "--root-dir",
        type=str,
        default=io_c.get("root_dir", DEFAULT_ROOT_DIR),
        help="Root dataset directory (must contain train/ and test/).",
    )
    io_g.add_argument(
        "--output-dir",
        type=str,
        default=io_c.get("output_dir", DEFAULT_OUTPUT_DIR),
        help="Directory to save model weights and plots.",
    )

    # --- Training ---
    train_g = parser.add_argument_group("Training")
    train_g.add_argument(
        "--num-epochs",
        type=int,
        default=train_c.get("num_epochs", DEFAULT_NUM_EPOCHS),
    )
    train_g.add_argument(
        "--batch-size",
        type=int,
        default=train_c.get("batch_size", DEFAULT_BATCH_SIZE),
    )
    train_g.add_argument(
        "--lr",
        type=float,
        default=train_c.get("lr", DEFAULT_LR),
    )
    train_g.add_argument(
        "--seed",
        type=int,
        default=train_c.get("seed", DEFAULT_SEED),
        help="Global RNG seed for reproducibility.",
    )

    # --- Loss ---
    # Note: Loss configuration is handled as a complex object in section 5
    # (parsed directly from YAML as a list of dicts, not via CLI arguments)

    # --- Optimizer ---
    optim_g = parser.add_argument_group("Optimizer")
    optim_g.add_argument(
        "--optimizer-type",
        type=str,
        default=optim_c.get("type", DEFAULT_OPTIMIZER_TYPE),
        help="Optimizer key: 'adam', 'adamw', 'sgd'.",
    )
    optim_g.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=train_c.get("grad_clip_max_norm", DEFAULT_GRAD_CLIP_MAX_NORM),
        help=(
            "Max norm for gradient clipping (torch.nn.utils.clip_grad_norm_). "
            "Set to null/None to disable."
        ),
    )

    # --- Scheduler ---
    sched_g = parser.add_argument_group("Scheduler")
    sched_g.add_argument(
        "--scheduler-patience",
        type=int,
        default=sched_c.get("patience", DEFAULT_SCHEDULER_PATIENCE),
    )
    sched_g.add_argument(
        "--scheduler-factor",
        type=float,
        default=sched_c.get("factor", DEFAULT_SCHEDULER_FACTOR),
    )
    sched_g.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=sched_c.get("min_lr", DEFAULT_SCHEDULER_MIN_LR),
    )

    # --- Cross Validation ---
    cv_g = parser.add_argument_group("Cross Validation")
    cv_g.add_argument(
        "--k-folds",
        type=int,
        default=cv_c.get("k_folds", DEFAULT_K_FOLDS),
    )

    # --- Early Stopping ---
    es_g = parser.add_argument_group("Early Stopping")
    es_g.add_argument(
        "--early-stopping-patience",
        type=int,
        default=es_c.get("patience", DEFAULT_EARLY_STOPPING_PATIENCE),
        help=(
            "Number of epochs without val_loss improvement before "
            "stopping. Set to 0 to disable early stopping."
        ),
    )
    es_g.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=es_c.get("min_delta", DEFAULT_EARLY_STOPPING_MIN_DELTA),
        help=(
            "Minimum improvement in monitored metric to count as "
            "progress. Smaller values are more sensitive."
        ),
    )
    es_g.add_argument(
        "--early-stopping-mode",
        type=str,
        default=es_c.get("mode", DEFAULT_EARLY_STOPPING_MODE),
        help=("'min' (lower is better) or 'max' (higher is better)."),
    )
    es_g.add_argument(
        "--early-stopping-monitor",
        type=str,
        default=es_c.get("monitor", DEFAULT_EARLY_STOPPING_MONITOR),
        help=(
            "Validation metric key to monitor for early stopping. "
            "Available: val_loss, val_mae_mean, val_rmse_mean, "
            "val_nae_mean, val_sre_mean, val_psnr_mean, val_ssim_mean."
        ),
    )

    # --- Model ---
    model_g = parser.add_argument_group("Model")
    model_g.add_argument(
        "--model-type",
        type=str,
        default=model_c.get("type", DEFAULT_MODEL_TYPE),
        help="Model key: 'plain_conv_unet', 'residual_encoder_unet'.",
    )
    model_g.add_argument(
        "--num-classes",
        type=int,
        default=model_c.get("num_classes", DEFAULT_NUM_CLASSES),
    )
    model_g.add_argument(
        "--input-channels",
        type=int,
        default=model_c.get("input_channels", DEFAULT_INPUT_CHANNELS),
    )
    model_g.add_argument(
        "--deep-supervision",
        type=_str2bool,
        default=model_c.get("deep_supervision", DEFAULT_DEEP_SUPERVISION),
    )

    # --- Augmentations ---
    aug_g = parser.add_argument_group("Augmentations")
    aug_g.add_argument(
        "--aug-level",
        type=str,
        default=aug_c.get("level", DEFAULT_AUG_LEVEL),
        help="Augmentation level: 'basic', 'medium', or 'full'.",
    )

    # --- Data ---
    data_g = parser.add_argument_group("Data")
    data_g.add_argument(
        "--img-size",
        type=int,
        default=data_c.get("img_size", DEFAULT_IMG_SIZE),
    )
    data_g.add_argument(
        "--class-names",
        nargs="+",
        default=data_c.get("class_names", DEFAULT_CLASS_NAMES),
        help="Display names for each class.",
    )
    data_g.add_argument(
        "--norm-mean",
        nargs="+",
        type=float,
        default=data_c.get("norm_mean", DEFAULT_NORM_MEAN),
    )
    data_g.add_argument(
        "--norm-std",
        nargs="+",
        type=float,
        default=data_c.get("norm_std", DEFAULT_NORM_STD),
    )
    data_g.add_argument(
        "--fill-value",
        type=int,
        default=data_c.get("fill_value", DEFAULT_FILL_VALUE),
    )
    data_g.add_argument(
        "--use-roi-mask",
        type=_str2bool,
        default=data_c.get("use_roi_mask", DEFAULT_USE_ROI_MASK),
        help="Focus loss and metrics on ROI regions only (filter out background).",
    )
    data_g.add_argument(
        "--channel-to-predict",
        type=int,
        default=data_c.get("channel_to_predict", None),
        help=(
            "Index of the channel to predict (0-based). When set and "
            "num_classes=1, the dataset loads only this channel from the "
            "multi-channel density maps. Ignored when num_classes > 1."
        ),
    )

    # --- WandB ---
    wb_g = parser.add_argument_group("Weights & Biases")
    wb_g.add_argument(
        "--use-wandb",
        type=_str2bool,
        default=wandb_c.get("enabled", DEFAULT_USE_WANDB),
        help="Enable WandB experiment tracking.",
    )
    wb_g.add_argument(
        "--wandb-project",
        type=str,
        default=wandb_c.get("project", DEFAULT_WANDB_PROJECT),
    )
    wb_g.add_argument(
        "--wandb-group",
        type=str,
        default=wandb_c.get("group", DEFAULT_WANDB_GROUP),
    )
    wb_g.add_argument(
        "--wandb-tags",
        nargs="+",
        default=wandb_c.get("tags", DEFAULT_WANDB_TAGS),
        help="Tags for the WandB run (e.g. 'baseline', 'foreground-loss').",
    )
    wb_g.add_argument(
        "--wandb-run-name",
        type=str,
        default=wandb_c.get("run_name", DEFAULT_WANDB_RUN_NAME),
        help=(
            "Custom WandB run name. If not set, auto-generated from "
            "the YAML config filename + timestamp."
        ),
    )

    # --- Debug ---
    parser.add_argument(
        "--debug",
        type=_str2bool,
        default=config.get("debug", DEFAULT_DEBUG),
        help="Enable DEBUG-level logging.",
    )

    # ---------------------------------------------------------
    # 4. Parse remaining CLI args
    # ---------------------------------------------------------
    args = parser.parse_args(remaining_argv)

    # Preserve the original --config path (eaten by pre-parse)
    args.config = known_args.config

    # ---------------------------------------------------------
    # 5. Inject complex objects that argparse cannot handle
    # ---------------------------------------------------------

    # Model architecture kwargs (dict from YAML, not parse-able via CLI)
    yaml_model_kwargs = model_c.get("kwargs", {})
    args.model_kwargs = {**DEFAULT_MODEL_KWARGS, **yaml_model_kwargs}

    # Loss configurations (list of dicts with 'type', 'weight', and optional custom args)
    # Each loss config must have 'type' and 'weight'; other keys depend on loss type
    if isinstance(loss_c, list):
        args.loss_configs = loss_c
    else:
        # Fallback for backward compatibility (if key-value structure is used)
        args.loss_configs = DEFAULT_LOSS_CONFIGS

    # Optimizer kwargs (dict from YAML, not parse-able via CLI)
    yaml_optim_kwargs = optim_c.get("kwargs", {})
    # Convert YAML list [b1, b2] → tuple for PyTorch betas param
    if "betas" in yaml_optim_kwargs and isinstance(yaml_optim_kwargs["betas"], list):
        yaml_optim_kwargs["betas"] = tuple(yaml_optim_kwargs["betas"])
    args.optimizer_kwargs = {**DEFAULT_OPTIMIZER_KWARGS, **yaml_optim_kwargs}

    # ---------------------------------------------------------
    # 6. Validation
    # ---------------------------------------------------------
    if args.root_dir is None:
        parser.error("--root-dir is required (via CLI or YAML IO.root_dir).")
    if args.output_dir is None:
        parser.error("--output-dir is required (via CLI or YAML IO.output_dir).")
    _validate_channel_selection(
        parser=parser,
        num_classes=args.num_classes,
        channel_to_predict=args.channel_to_predict,
        class_names=list(args.class_names),
    )

    # ---------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------
    logger.info(f"\n{'=' * 10} CONFIG LOADED {'=' * 10}")
    logger.info(f"\n{pprint.pformat(vars(args))}")
    logger.info(f"{'=' * 35}\n")

    return args
