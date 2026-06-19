"""WandB tracking callback for the density-estimation trainer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class TrackingCallback:
    """
    Minimal interface for experiment-tracking backends (WandB, …).

    Override any method you need; the defaults are no-ops.
    """

    def on_epoch_end(
        self,
        phase: str,
        epoch: int,
        metrics: Dict[str, float],
        fold: int | None = None,
    ) -> None: ...

    def on_fold_end(self, fold: int, metrics: Dict[str, Any]) -> None: ...

    def on_cv_end(self, cv_summary: Dict[str, Any]) -> None: ...

    def log_cv_epoch_averages(self, epoch_averages: Dict[str, list]) -> None:
        """Log per-epoch cross-fold averaged metrics (for WandB charts)."""
        ...

    def on_training_end(self, final_metrics: Dict[str, Any]) -> None: ...

    def log_artifact(self, path: str, name: str) -> None: ...


class NoOpCallback(TrackingCallback):
    """Default callback that does nothing."""


class WandBCallback(TrackingCallback):
    """
    Logs training metrics and artifacts to Weights & Biases.

    If ``enabled=False`` (or wandb is not installed), all calls are silent no-ops.

    Args:
        enabled: Toggle tracking on/off.
        project: WandB project name.
        group: WandB run group.
        config: Dict of hyperparameters to log.
        tags: List of string tags for the run.
        run_name: Explicit WandB run name.  When ``None`` an automatic
            name is generated from the YAML config filename + timestamp.
        config_path: Path to the YAML config file.  Used to derive a
            human-readable run name when *run_name* is ``None``.
        existing_run: If provided, reuse this ``wandb.Run`` instead of
            calling ``wandb.init()``. Used by the sweep agent where the
            sweep controller has already initialised a run.
    """

    def __init__(
        self,
        enabled: bool = True,
        project: str = "neuro_brain_project",
        group: str = "density_estimator",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        run_name: Optional[str] = None,
        config_path: Optional[str] = None,
        existing_run: Optional[Any] = None,
    ):
        self.enabled = enabled and wandb is not None
        self.run = None
        self._owns_run = False  # whether we created the run (and should finish it)

        if not self.enabled:
            if enabled and wandb is None:
                logger.warning(
                    "wandb is not installed — tracking disabled. "
                    "Install with: pip install wandb"
                )
            logger.info("WandB tracking disabled.")
            return

        # --- Derive a human-readable run name ---
        if run_name is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            if config_path:
                from pathlib import Path as _P

                yaml_stem = _P(config_path).stem
                run_name = f"{yaml_stem}_{timestamp}"
            else:
                run_name = f"density_{timestamp}"

        if existing_run is not None:
            # Reuse sweep-managed run
            self.run = existing_run
            self._owns_run = False
            # Update config on the existing run
            if config:
                self.run.config.update(config, allow_val_change=True)
            logger.info(f"WandB callback reusing existing run: {self.run.name}")
        else:
            self.run = wandb.init(
                project=project,
                group=group,
                name=run_name,
                config=config or {},
                tags=tags or [],
                job_type="training",
            )
            self._owns_run = True
            logger.info(f"WandB run initialised: {self.run.name}")

    # --- TrackingCallback interface ---

    def on_epoch_end(
        self,
        phase: str,
        epoch: int,
        metrics: Dict[str, Any],
        fold: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        prefix = f"fold_{fold}/" if fold is not None else ""

        parsed_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                parsed_metrics[k] = v
            elif hasattr(v, "item"):
                val = v.item() if v.ndim == 0 else v.mean().item()
                parsed_metrics[k] = val

        wandb.log({f"{prefix}{phase}/{k}": v for k, v in parsed_metrics.items()})

    def on_fold_end(self, fold: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        # Log a summary scalar for the fold
        for key in ("val_loss", "val_mae"):
            if key in metrics:
                metric_val = metrics[key]

                if not isinstance(metric_val, (int, float)):
                    metric_val = float(np.array(metric_val).mean())

                wandb.run.summary[f"fold_{fold}/{key}_best"] = metric_val

    def log_cv_epoch_averages(self, epoch_averages: Dict[str, list]) -> None:
        """Log per-epoch cross-fold averaged metrics as WandB line charts.

        Args:
            epoch_averages: ``{metric_name: [val_epoch_0, val_epoch_1, …]}``.
                Keys like ``train_loss``, ``val_loss``, ``val_nae_mean``, etc.
        """
        if not self.enabled:
            return
        if not epoch_averages:
            return

        # Determine the number of epochs from any key
        n_epochs = len(next(iter(epoch_averages.values())))
        for epoch in range(n_epochs):
            row = {f"cv_avg/{k}": vals[epoch] for k, vals in epoch_averages.items()}
            row["cv_epoch"] = epoch
            wandb.log(row)

        logger.info(
            f"Logged {n_epochs} epoch-averaged CV metrics to WandB "
            f"(keys: {list(epoch_averages.keys())})"
        )

    def on_cv_end(self, cv_summary: Dict[str, Any]) -> None:
        """
        Log cross-validation summary to WandB summary scalars.

        """
        if not self.enabled:
            return
        for key, val in cv_summary.items():
            if not isinstance(val, dict):
                continue

            if "mean" in val:
                wandb.run.summary[f"cv/{key}_mean"] = val["mean"]
            if "std" in val:
                wandb.run.summary[f"cv/{key}_std"] = val["std"]

    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Log all final metrics to WandB summary and as logged scalars.

        Handles scalars, lists (per-class metrics), and nested dicts
        (e.g. GAME with levels → {"mean": [...], "std": [...]}).
        Add or remove entries from *final_metrics* in the trainer and
        they will be logged here automatically — no callback changes needed.
        """
        if not self.enabled:
            return

        def _to_plain(v: Any) -> Any:
            """Convert numpy/tensor values to plain Python types."""
            if hasattr(v, "tolist"):
                return v.tolist()
            return v

        log_dict: Dict[str, Any] = {}

        for name, val in final_metrics.items():
            if isinstance(val, dict):
                # Nested dict (e.g. GAME: {"0": {"mean": [...], "std": [...]}})
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict):
                        for stat, stat_val in sub_val.items():
                            key = f"test/{name}_{sub_key}_{stat}"
                            plain = _to_plain(stat_val)
                            wandb.run.summary[key] = plain
                            log_dict[key] = plain
                    else:
                        key = f"test/{name}_{sub_key}"
                        plain = _to_plain(sub_val)
                        wandb.run.summary[key] = plain
                        log_dict[key] = plain
            elif isinstance(val, list):
                # Per-class list (e.g. MAE: [0.1, 0.2, 0.3])
                wandb.run.summary[f"test/{name}"] = val
                mean_val = sum(val) / len(val) if val else 0.0
                log_dict[f"test/{name}_mean"] = mean_val
                wandb.run.summary[f"test/{name}_mean"] = mean_val
            else:
                plain = _to_plain(val)
                wandb.run.summary[f"test/{name}"] = plain
                log_dict[f"test/{name}"] = plain

        # Also log as step-data so metrics appear in wandb charts
        if log_dict:
            wandb.log(log_dict)

        if self._owns_run:
            wandb.finish()
            logger.info("WandB run finished.")
        else:
            logger.info("WandB final metrics logged (run managed externally).")

    def log_artifact(self, path: str, name: str) -> None:
        # Model artifacts are already saved locally; skip uploading to WandB
        # to avoid unnecessary storage usage.
        pass
