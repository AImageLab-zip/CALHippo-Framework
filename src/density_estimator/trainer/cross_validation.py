import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from src.density_estimator.datasets.density_dataset import get_groups
from src.density_estimator.models import reset_model_weights
from src.density_estimator.optimizers import build_optimizer
from src.density_estimator.tracking.tracking import TrackingCallback
from src.density_estimator.trainer.early_stopping import EarlyStopping
from src.density_estimator.trainer.evaluate import (
    evaluate_model_on_loader,
    get_evaluated_metrics_list,
)
from src.density_estimator.trainer.helpers import get_dataloader_num_workers
from src.density_estimator.trainer.metrics_aggregation import (
    aggregate_best_epoch_metrics,
    compute_average_metrics,
)
from src.density_estimator.trainer.single_epoch_train import train_one_epoch
from src.density_estimator.utils.reproducibility import seed_worker
from src.density_estimator.utils.visualization import plot_cv_losses, plot_cv_metrics


def train_cross_validation(
    args: Namespace,
    k_folds: int,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    criterion: torch.nn.Module,
    device: torch.device,
    opt_type: str,
    opt_kwargs: dict,
    es_monitor: str,
    es_mode: str,
    grad_clip: float | None,
    class_names: List[str],
    use_roi_mask: bool,
    patch_size_out: int,
    use_log_counts: bool,
    callback: TrackingCallback,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[int]]:
    """
    Perform grouped K-Fold cross-validation and return the CV history and summary.
    """

    # Dataset Split: Grouped K-Fold CV (by WSI)
    groups = get_groups(train_dataset)
    dummy_X = np.zeros(len(train_dataset))

    if k_folds <= 1:
        # Single grouped 80/20 train/val split
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        fold_iterator = list(enumerate(splitter.split(dummy_X, groups=groups)))
        n_folds = 1
        logger.info("Using single grouped 80/20 train/val split (k_folds=1)")
    else:
        gkf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        fold_iterator = list(enumerate(gkf.split(dummy_X, groups=groups)))
        n_folds = k_folds

    # CV history initialisation
    train_loss_comp = [f"train_loss_c_{c[0]}" for c in criterion.losses]
    validation_metrics = [f"val_{m}" for m in get_evaluated_metrics_list(criterion)]
    all_evaluated_metrics = ["train_loss"] + train_loss_comp + validation_metrics

    cv_folds_best_epoch_metrics: List[
        Dict[str, Any]
    ] = []  # Best epoch metrics dict for each fold
    cv_folds_best_epoch: List[int] = []  # Best epoch numbers for each fold

    cv_history: Dict[str, List] = {
        k: [] for k in all_evaluated_metrics
    }  # for example {"val_mae": [list of per-epoch arrays of shape (C,)], "val_nae": [...], ...}

    # Workers per loader: during CV, train + val loaders are active together
    cv_workers = get_dataloader_num_workers(num_loaders=2)
    logger.info(f"Starting {n_folds}-Fold Group CV … (num_workers={cv_workers}/loader)")

    # Cross-validation loop
    for fold, (train_idx, val_idx) in fold_iterator:
        logger.info(f"Fold {fold + 1}/{n_folds}")

        # Leak check
        train_wsis = set(groups[train_idx])
        val_wsis = set(groups[val_idx])
        assert train_wsis.isdisjoint(val_wsis), "WSI leakage detected!"
        logger.info(f"  Train WSIs: {len(train_wsis)} | Val WSIs: {len(val_wsis)}")

        # Reset model & build optimizer
        model.apply(reset_model_weights)
        optimizer = build_optimizer(
            opt_type, model.parameters(), lr=args.lr, **opt_kwargs
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )

        # Dataloaders initialisation with fold-specific samplers
        g_train = torch.Generator()
        g_train.manual_seed(args.seed + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx, generator=g_train),
            pin_memory=True,
            persistent_workers=cv_workers > 0,
            num_workers=cv_workers,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g_train,
        )
        g_val = torch.Generator()
        g_val.manual_seed(args.seed + fold)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_idx, generator=g_val),
            num_workers=cv_workers,
            worker_init_fn=seed_worker,
            generator=g_val,
        )

        es = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=es_mode,
        )

        # List of metric dicts for each epoch
        current_fold_history: List[Dict[str, Any]] = []

        # Fold loop (epoch-level)
        epoch_bar = tqdm(range(args.num_epochs), desc=f"Fold {fold + 1}", position=0)
        for epoch in epoch_bar:
            train_loss, loss_components = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_roi_mask=use_roi_mask,
                grad_clip_max_norm=grad_clip,
                patch_size_out=patch_size_out,
                use_log_counts=use_log_counts,
            )

            val_metrics = evaluate_model_on_loader(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_roi_mask=use_roi_mask,
                patch_size_out=patch_size_out,
                use_log_counts=use_log_counts,
            )

            scheduler.step(val_metrics["loss"])

            # Merge metrics into a single dict
            current_epoch_metrics = {}
            current_epoch_metrics["train_loss"] = train_loss
            for c, lc in loss_components.items():
                current_epoch_metrics[f"train_loss_c_{c}"] = lc
            for k in val_metrics.keys():
                key_name = f"val_{k}"
                current_epoch_metrics[key_name] = val_metrics[k]

            # Log metrics to fold history and on WandB
            current_fold_history.append(current_epoch_metrics)

            epoch_bar.set_postfix(
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                loss=f"T:{train_loss:.3f}/V:{val_metrics['loss']:.3f}",
                nae=f"{float(np.mean(val_metrics['nae'])):.3f}",
                nae_classes=", ".join(f"{v:.3f}" for v in val_metrics["nae"]),
                ssim=f"{float(np.mean(val_metrics['ssim'])):.3f}",
            )

            callback.on_epoch_end(
                phase="cv",
                epoch=epoch,
                metrics=current_epoch_metrics,
                fold=fold,
            )

            # Early stopping
            es_value = current_epoch_metrics.get(es_monitor)
            if not isinstance(es_value, (float, int)):
                es_value = float(np.mean(es_value))

            es.step(es_value, epoch, model)
            if es.should_stop:
                logger.info(
                    f"  Early stopping triggered at epoch {epoch + 1} "
                    f"(best epoch {es.best_epoch + 1}, "
                    f"best {es_monitor} {es.best_score:.4f})"
                )
                es.restore_best_weights(model)
                cv_folds_best_epoch_metrics.append(current_fold_history[es.best_epoch])
                cv_folds_best_epoch.append(es.best_epoch)
                break
        else:
            # If we didn't break from the loop, save the last epoch's metrics
            cv_folds_best_epoch_metrics.append(current_fold_history[-1])
            cv_folds_best_epoch.append(args.num_epochs - 1)

        # Log fold metrics to WandB
        current_fold_best = cv_folds_best_epoch_metrics[
            -1
        ]  # Metrics dict of the best epoch for the current fold
        callback.on_fold_end(fold=fold, metrics=current_fold_best)

        # Save fold history to CV history
        for k in cv_history.keys():
            # Extract the metric values across epochs for the current fold
            fold_metric_values = [
                epoch_metrics[k] for epoch_metrics in current_fold_history
            ]
            cv_history[k].append(fold_metric_values)

    # ---- CV summary with mean ± std ----
    cv_summary = aggregate_best_epoch_metrics(
        folds_best_metrics=cv_folds_best_epoch_metrics,
        class_names=class_names,
        log_metrics=True,
    )
    callback.on_cv_end(cv_summary=cv_summary)

    # ---- Log cross-fold averaged epoch curves to WandB ----
    epoch_averages = compute_average_metrics(cv_history)
    callback.log_cv_epoch_averages(epoch_averages)

    # ---- Save CV plots ----
    os.makedirs(args.output_dir, exist_ok=True)
    plot_cv_losses(
        cv_history,
        save_path=os.path.join(args.output_dir, "cv_losses.png"),
    )
    plot_cv_metrics(
        cv_history,
        save_path=os.path.join(args.output_dir, "cv_metrics.png"),
    )

    return cv_history, cv_summary, cv_folds_best_epoch
