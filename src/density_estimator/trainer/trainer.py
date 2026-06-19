"""Training orchestrator: GroupKFold CV → final retrain → test evaluation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.density_estimator.datasets.density_dataset import (
    SimpleCADataset,
    get_transforms,
)
from src.density_estimator.losses import build_loss
from src.density_estimator.models import build_model
from src.density_estimator.tracking import NoOpCallback, TrackingCallback
from src.density_estimator.trainer.cross_validation import train_cross_validation
from src.density_estimator.trainer.evaluate import evaluate_model_on_loader
from src.density_estimator.trainer.final_training import final_training
from src.density_estimator.trainer.helpers import get_dataloader_num_workers
from src.density_estimator.utils.reproducibility import seed_worker

# ---------------------------------------------------------------------------
# Summary CSV export
# ---------------------------------------------------------------------------


def save_summary_json(
    cv_summary: Dict[str, Any],
    cv_best_epochs: List[int],
    final_metrics: Dict[str, Any],
    save_path: str,
) -> None:
    summary = {
        "final_metrics": final_metrics,
        "cv_best_metrics": cv_summary,
        "cv_best_epochs": cv_best_epochs,
    }

    # ---- Write ----
    with open(save_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary JSON saved → {save_path}")


def save_fold_metrics_json(
    cv_fold_best_metrics: List[Dict[str, Any]],
    save_path: str,
) -> None:
    with open(save_path, "w") as fh:
        json.dump(cv_fold_best_metrics, fh, indent=2)
    logger.info(f"CV fold metrics JSON saved → {save_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_cross_validation(
    args, callback: TrackingCallback | None = None
) -> Dict[str, Any]:
    """
    Execute GroupKFold cross-validation and final retrain.

    Args:
        args: Parsed CLI namespace (see ``config.py``).
        callback: Optional tracking callback for logging metrics.

    Returns:
        Dict with keys:
            ``cv_history``, ``cv_summary``, ``final_train_history``,
            ``final_metrics``, ``model_path``.
    """

    # ==================================================================
    # Setup: device, datasets, model, loss, optimizer, etc.
    # ==================================================================

    cb = callback or NoOpCallback()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ---- ROI mask flag ----
    use_roi_mask = getattr(args, "use_roi_mask", False)
    logger.info(f"ROI mask filtering: {'enabled' if use_roi_mask else 'disabled'}")

    # ---- Channel to predict (for single-channel mode) ----
    channel_to_predict = getattr(args, "channel_to_predict", None)
    if channel_to_predict is not None:
        logger.info(
            f"Single-channel mode: predicting channel {channel_to_predict} only"
        )

    # ---- Class Names ----
    class_names = list(getattr(args, "class_names", ["C0", "C1", "C2"]))
    if args.num_classes == 1:
        if channel_to_predict is None:
            raise ValueError("channel_to_predict must be set when num_classes=1.")
        if not 0 <= channel_to_predict < len(class_names):
            raise ValueError(
                f"channel_to_predict={channel_to_predict} is out of range for class_names={class_names}."
            )
        class_names = [class_names[channel_to_predict]]
    logger.info(f"Class names: {class_names}")

    # ---- transforms ----
    aug_level = getattr(args, "aug_level", "basic")
    train_tf, val_tf = get_transforms(
        img_size=args.img_size,
        norm_mean=tuple(args.norm_mean),
        norm_std=tuple(args.norm_std),
        aug_level=aug_level,
        load_roi_masks=use_roi_mask,
    )
    logger.info(f"Augmentation level: {aug_level}")

    # ---- dataset ----
    train_dataset = SimpleCADataset(
        root_dir=args.root_dir,
        split="train",
        transform=train_tf,
        max_pix_value=float(args.fill_value),
        load_roi_masks=use_roi_mask,
        channel_to_predict=channel_to_predict,
    )
    # Separate dataset instance with deterministic val transforms for CV
    val_dataset = SimpleCADataset(
        root_dir=args.root_dir,
        split="train",
        transform=val_tf,
        max_pix_value=float(args.fill_value),
        load_roi_masks=use_roi_mask,
        channel_to_predict=channel_to_predict,
    )
    logger.info(f"Dataset: {len(train_dataset)} patches")

    # ---- loss (via factory) ----
    criterion = build_loss(args.loss_configs).to(device)
    logger.info(f"Loss created with config: {args.loss_configs}")

    # ---- model (via factory, created once, reset per fold) ----
    model_kwargs = getattr(args, "model_kwargs", {})
    model = build_model(
        model_type=args.model_type,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        deep_supervision=args.deep_supervision,
        **model_kwargs,
    ).to(device)
    logger.info(f"Model: {args.model_type} → {model.__class__.__name__}")

    # ---- patched output config ----
    patch_size_out: int = int(model_kwargs.get("patch_size_out", 1))
    use_log_counts: bool = bool(model_kwargs.get("use_log_counts", False))
    if patch_size_out > 1:
        target_mode = "log1p-compressed" if use_log_counts else "raw counts"
        logger.info(
            f"Patched density mode: patch_size_out={patch_size_out}  "
            f"→  output grid {args.img_size // patch_size_out}×{args.img_size // patch_size_out}  "
            f"(targets: {target_mode})"
        )
        if use_roi_mask:
            raise ValueError(
                "use_roi_mask=True is not yet compatible with patch_size_out > 1. "
                "The ROI mask has full spatial resolution while predictions are "
                f"{args.img_size // patch_size_out}×{args.img_size // patch_size_out}."
            )

    # ---- optimizer config ----
    opt_type = getattr(args, "optimizer_type", "adam")
    opt_kwargs = getattr(args, "optimizer_kwargs", {})
    logger.info(f"Optimizer: {opt_type} (kwargs={opt_kwargs})")

    # ---- gradient clipping config ----
    grad_clip = getattr(args, "grad_clip_max_norm", None)
    logger.info(
        f"Gradient clipping: {'max_norm=' + str(grad_clip) if grad_clip else 'disabled'}"
    )

    # ---- early stopping config ----
    es_monitor = getattr(args, "early_stopping_monitor", "val_nae")
    es_mode = getattr(args, "early_stopping_mode", "min")
    logger.info(
        f"Early stopping: monitor={es_monitor}, mode={es_mode} "
        f"(patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta})"
    )

    # ==================================================================
    # Cross-validation (or single grouped split when k_folds == 1)
    # ==================================================================

    cv_history, cv_summary, cv_best_epochs = train_cross_validation(
        args=args,
        k_folds=args.k_folds,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        device=device,
        opt_type=opt_type,
        opt_kwargs=opt_kwargs,
        es_monitor=es_monitor,
        es_mode=es_mode,
        grad_clip=grad_clip,
        class_names=class_names,
        use_roi_mask=use_roi_mask,
        patch_size_out=patch_size_out,
        use_log_counts=use_log_counts,
        callback=cb,
    )

    # ==================================================================
    # Final retrain on full training set
    # ==================================================================

    test_dataset = SimpleCADataset(
        root_dir=args.root_dir,
        split="test",
        transform=val_tf,
        max_pix_value=float(args.fill_value),
        load_roi_masks=use_roi_mask,
        channel_to_predict=channel_to_predict,
    )

    # Final training epoch numbers based on average best epoch from CV (with a +10% margin)
    # The es.best_epoch is 0-indexed, so we add 1 for epoch count
    avg_best_epoch = float(np.mean(cv_best_epochs))
    final_epochs = int((avg_best_epoch + 1) * 1.1)
    logger.info(
        f"Final retrain epochs: {final_epochs} (average CV best epoch: {avg_best_epoch + 1:.1f} + 10% margin)"
    )

    final_train_history = final_training(
        args=args,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        criterion=criterion,
        epochs=final_epochs,
        device=device,
        opt_type=opt_type,
        opt_kwargs=opt_kwargs,
        grad_clip=grad_clip,
        use_roi_mask=use_roi_mask,
        patch_size_out=patch_size_out,
        use_log_counts=use_log_counts,
        callback=cb,
    )

    # ==================================================================
    # Final evaluation on test set
    # ==================================================================
    logger.info("Computing final metrics on test set …")

    g_test = torch.Generator()
    g_test.manual_seed(args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=get_dataloader_num_workers(num_loaders=1),
        generator=g_test,
    )

    final_metrics = evaluate_model_on_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_roi_mask=use_roi_mask,
        patch_size_out=patch_size_out,
        use_log_counts=use_log_counts,
    )

    # Print report
    logger.info("=" * 40)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("=" * 40)

    ordered_keys = sorted(final_metrics.keys())
    for key in ordered_keys:
        value = final_metrics[key]
        if isinstance(value, list):
            mean_val = np.mean(value)
            class_str = ", ".join(
                f"{name} {v:.4f}" for name, v in zip(class_names, value)
            )
            logger.info(f"  {key:20s}:  {mean_val:.4f} {class_str}")
        else:
            logger.info(f"  {key:20s}:  {value:.4f}")

    logger.info("=" * 40)

    # ---- Save summary JSON ----
    save_summary_json(
        cv_summary=cv_summary,
        cv_best_epochs=cv_best_epochs,
        final_metrics=final_metrics,
        save_path=os.path.join(args.output_dir, "summary_metrics.json"),
    )

    # Save model checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "final_density_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    cb.log_artifact(model_path, "final_density_model")

    cb.on_training_end(final_metrics=final_metrics)

    return {
        "cv_history": cv_history,
        "cv_summary": cv_summary,
        "final_train_history": final_train_history,
        "final_metrics": final_metrics,
        "model_path": model_path,
        "model": model,
        "test_dataset": test_dataset,
        "device": device,
        "use_roi_mask": use_roi_mask,
        "patch_size_out": patch_size_out,
        "use_log_counts": use_log_counts,
    }
