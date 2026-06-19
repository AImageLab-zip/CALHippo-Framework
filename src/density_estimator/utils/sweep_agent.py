"""
W&B Sweep Agent entry point for the density-estimation pipeline.

This script bridges the W&B Sweep system with the existing training pipeline.
The sweep agent calls this script for each trial.  It:

1. Reads hyperparameters from ``wandb.config`` (set by the sweep controller).
2. Maps them to the CLI ``argv`` expected by ``src.density_estimator.train``.
3. Patches the WandB callback so the **existing** sweep-managed ``wandb.run``
   is reused (no duplicate ``wandb.init()``).
4. Logs the CV mean val_loss to ``wandb.run.summary`` so the Bayesian
   controller can read it.

Usage (called automatically by ``wandb agent``):
    python -m src.density_estimator.utils.sweep_agent <args_json_file>

Or via the helper ``launch_sweep()`` function for programmatic use.
"""

from __future__ import annotations

import os

# Increase timeout for W&B service startup in slow environments (e.g. Slurm)
os.environ["WANDB_SERVICE_WAIT"] = "300"

from typing import Any, Dict, List, Optional

from loguru import logger

# Keys in wandb.config that are NOT simple CLI flags and need
# special handling (or should be skipped entirely).
_SKIP_KEYS = frozenset({"config", "model_kwargs"})


def _build_argv_from_wandb_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Convert wandb.config dict → CLI argv list for ``get_args()``.

    Convention: every key in the sweep YAML ``parameters`` section must
    match the **argparse dest name** used in ``config.py``
    (e.g. ``batch_size``, ``optimizer_type``, ``scheduler_min_lr``).
    The CLI flag is derived automatically::

        batch_size  →  --batch-size
        lr          →  --lr

    Keys listed in ``_SKIP_KEYS`` are handled separately (or ignored).
    To sweep a new scalar parameter, just add it to the sweep YAML —
    no code changes needed here.
    """
    argv: List[str] = []

    # 1. Base config YAML (always present)
    if "config" in cfg:
        argv += ["--config", str(cfg["config"])]

    # 2. Auto-derive CLI flags from every other key
    for key, value in cfg.items():
        if key in _SKIP_KEYS:
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        argv += [cli_flag, str(value)]

    # 3. Force wandb on (sweep runs always tracked)
    argv += ["--use-wandb", "true"]

    return argv


def run_sweep_trial() -> None:
    """
    Execute a single sweep trial.

    Called once per run by the sweep agent.  The flow is:

    1. ``wandb.init()`` is called HERE (by us) so the sweep controller
       can inject its chosen hyperparameters into ``wandb.config``.
    2. We read ``wandb.config`` and convert it to CLI args.
    3. We call the existing ``main()`` pipeline with those args.
    4. We patch the WandB callback to REUSE the already-active run.
    5. After training, we log the sweep's target metric
       (``cv_mean_val_loss``) to ``wandb.run.summary``.
    """
    import wandb

    # --- 1. Init (the sweep agent injects config here) ---
    run = wandb.init()
    cfg = dict(wandb.config)
    logger.info(f"Sweep trial started — run: {run.name}")
    logger.info(f"Sweep config: {cfg}")

    # --- 2. Build argv from sweep config ---
    argv = _build_argv_from_wandb_config(cfg)
    logger.info(f"Constructed argv: {argv}")

    # --- 3. Import and prepare ---
    import os
    import shutil
    from pathlib import Path

    from src.density_estimator.config import get_args
    from src.density_estimator.tracking.tracking import WandBCallback
    from src.density_estimator.trainer.trainer import run_cross_validation
    from src.density_estimator.utils.visualization import (
        plot_prediction_per_class,
        plot_prediction_summary,
    )
    from src.utils.helpers import (
        build_run_info,
        cv_history_to_serialisable,
        resolve_output_dir,
        save_json,
    )
    from src.utils.logger_setup import setup_logging

    args = get_args(argv)

    # --- 3b. Override model_kwargs if provided by sweep ---
    if "model_kwargs" in cfg:
        args.model_kwargs = dict(cfg["model_kwargs"])
        logger.info(f"Sweep model_kwargs: {args.model_kwargs}")

    # --- 4. Resolve output dir ---
    config_path = getattr(args, "config", None)
    run_dir = resolve_output_dir(args.output_dir, config_path)
    args.output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # --- 5. Logging ---
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

    # Copy source YAML
    if config_path and os.path.isfile(config_path):
        dest = os.path.join(run_dir, Path(config_path).name)
        shutil.copy2(config_path, dest)

    # Save run info
    run_info = build_run_info(args)
    save_json(run_info, os.path.join(run_dir, "run_info.json"), "Run info")

    # --- 6. Create callback that REUSES the sweep run ---
    callback = WandBCallback(
        enabled=True,
        project=args.wandb_project,
        group=args.wandb_group,
        config=vars(args),
        tags=getattr(args, "wandb_tags", []),
        config_path=config_path,
        existing_run=run,  # <-- key: reuse sweep-managed run
    )

    # --- 7. Train ---
    results = run_cross_validation(args, callback=callback)

    # --- 8. Log sweep metric ---
    # The Bayesian optimizer needs cv_best_mean_nae in summary
    cv_summary = results.get("cv_summary", {})

    # Best-epoch metrics (min MAE/NAE, max PSNR/SSIM averaged across folds)
    for best_key in (
        "best_mean_mae",
        "best_mean_nae",
        "best_mean_psnr",
        "best_mean_ssim",
    ):
        val = cv_summary.get(best_key, {}).get("mean")
        if val is not None:
            wandb_key = f"cv_{best_key}"
            wandb.run.summary[wandb_key] = val
            wandb.log({wandb_key: val})
            logger.info(f"Logged {wandb_key}={val:.6f} to sweep")

    # Also keep cv_mean_val_loss for backward compatibility
    cv_val_loss = cv_summary.get("val_loss", {})
    mean_val_loss = cv_val_loss.get("mean", float("inf"))
    wandb.run.summary["cv_mean_val_loss"] = mean_val_loss
    wandb.log({"cv_mean_val_loss": mean_val_loss})
    logger.info(f"Logged cv_mean_val_loss={mean_val_loss:.6f} to sweep")

    # --- 9. Save artifacts ---
    metrics_payload = {
        "cv_summary": results["cv_summary"],
        "cv_history": cv_history_to_serialisable(results["cv_history"]),
        "final_train_history": results["final_train_history"],
        "final_test_metrics": results["final_metrics"],
    }
    save_json(metrics_payload, os.path.join(run_dir, "metrics.json"), "Metrics")

    # NOTE: Final-train plots (final_train_losses.png, final_train_metrics.png)
    #       are saved inside the trainer.

    viz_kwargs = {
        "model": results["model"],
        "dataset": results["test_dataset"],
        "device": results["device"],
        "num_samples": 5,
        "class_names": getattr(args, "class_names", None),
        "mean_list": list(args.norm_mean),
        "std_list": list(args.norm_std),
        "gain": 150.0,
    }
    plot_prediction_summary(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "predictions_summary.png"),
    )
    plot_prediction_per_class(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "predictions_per_class.png"),
    )

    # --- 10. Finish ---
    # Don't call wandb.finish() here — the sweep agent handles it.
    # on_training_end is already called inside run_cross_validation(),
    # so we do NOT call it again here (would duplicate wandb.log data).

    logger.info(f"Sweep trial complete. Artifacts in → {run_dir}")


def launch_sweep(
    sweep_config_path: str,
    count: int = 50,
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """
    Programmatic helper to create a sweep and start the agent.

    Args:
        sweep_config_path: Path to the sweep YAML file.
        count: Number of runs to execute.
        project: W&B project name (overrides the one in sweep YAML).
        entity: W&B entity (team/user).

    Returns:
        The sweep ID.
    """
    import yaml

    import wandb

    with open(sweep_config_path) as fh:
        sweep_cfg = yaml.safe_load(fh)

    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=project or "neuro_brain_project",
        entity=entity,
    )
    logger.info(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=count)

    return sweep_id


# ---------------------------------------------------------------
# CLI entry point (called by wandb agent via 'command' config)
# ---------------------------------------------------------------
if __name__ == "__main__":
    # When wandb agent calls this with ${args_json_file},
    # the hyperparams are passed as a JSON file path.
    # wandb.init() reads them automatically.
    run_sweep_trial()
