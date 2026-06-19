"""
Entry point for the density-estimation training pipeline.

Usage::

    # With a YAML config (recommended):
    CONFIG=experiments/density_estimation/best_model/<CONFIG_FILE>.yaml
    python -m src.density_estimator --config "$CONFIG"

    # Override individual values via CLI:
    python -m src.density_estimator --config "$CONFIG" \\
        --lr 5e-4 --batch-size 32

    # Purely from CLI (no YAML):
    python -m src.density_estimator \\
        --root-dir /path/to/dataset \\
        --output-dir data/density_estimator_training
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from loguru import logger

from src.density_estimator.config import get_args
from src.density_estimator.tracking.tracking import WandBCallback
from src.density_estimator.trainer.trainer import run_cross_validation
from src.density_estimator.utils.visualization import (
    plot_prediction_per_class,
    plot_prediction_summary,
)
from src.utils.helpers import build_run_info, resolve_output_dir, save_json
from src.utils.logger_setup import setup_logging

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)

    # --- Resolve output directory ---
    config_path = getattr(args, "config", None)
    run_dir = resolve_output_dir(args.output_dir, config_path)
    args.output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # --- Logging: stderr + file sink in run directory ---
    setup_logging(debug=args.debug)
    log_path = os.path.join(run_dir, "run.log")
    logger.add(
        log_path,
        level="INFO" if not args.debug else "DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=args.debug,
    )
    logger.info(
        f"Density-estimation training pipeline starting …\n  Run directory: {run_dir}"
    )
    logger.debug(f"Args: {vars(args)}")

    # --- Copy source YAML config into run directory ---
    if config_path and os.path.isfile(config_path):
        dest = os.path.join(run_dir, Path(config_path).name)
        shutil.copy2(config_path, dest)
        logger.info(f"Config YAML copied → {dest}")

    # --- Save run_info.json (args + env metadata) ---
    run_info = build_run_info(args)
    save_json(run_info, os.path.join(run_dir, "run_info.json"), "Run info")

    # --- Tracking ---
    callback = WandBCallback(
        enabled=args.use_wandb,
        project=args.wandb_project,
        group=args.wandb_group,
        config=vars(args),
        tags=getattr(args, "wandb_tags", []),
        run_name=getattr(args, "wandb_run_name", None),
        config_path=config_path,
    )

    # --- Run training pipeline ---
    results = run_cross_validation(args, callback=callback)

    # --- Prediction visualizations (saved only, never displayed) ---
    # Adjust class names/colors for single-channel mode
    _ALL_CLASS_COLORS = ["red", "cyan", "blue"]
    channel_to_predict = getattr(args, "channel_to_predict", None)
    if args.num_classes > 1:
        channel_to_predict = None
    viz_class_names = getattr(args, "class_names", None)
    viz_class_colors = None  # let viz function use defaults
    if channel_to_predict is not None and viz_class_names:
        if channel_to_predict < len(viz_class_names):
            viz_class_names = [viz_class_names[channel_to_predict]]
        if channel_to_predict < len(_ALL_CLASS_COLORS):
            viz_class_colors = [_ALL_CLASS_COLORS[channel_to_predict]]

    viz_kwargs = {
        "model": results["model"],
        "dataset": results["test_dataset"],
        "device": results["device"],
        "num_samples": 5,
        "class_names": viz_class_names,
        "class_colors": viz_class_colors,
        "mean_list": list(args.norm_mean),
        "std_list": list(args.norm_std),
        "gain": 150.0,
        "show_roi_mask": results.get("use_roi_mask", False),
        "patch_size_out": results.get("patch_size_out", 1),
        "use_log_counts": results.get("use_log_counts", False),
    }
    plot_prediction_summary(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "predictions_summary.png"),
    )
    plot_prediction_per_class(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "predictions_per_class.png"),
    )

    logger.info(f"Pipeline complete. All artifacts in → {run_dir}")


if __name__ == "__main__":
    main()
