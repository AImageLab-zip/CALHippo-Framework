import os
from argparse import Namespace
from typing import Any, Dict, List

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.density_estimator.models import reset_model_weights
from src.density_estimator.optimizers import build_optimizer
from src.density_estimator.tracking.tracking import TrackingCallback
from src.density_estimator.trainer.evaluate import (
    evaluate_model_on_loader,
    get_evaluated_metrics_list,
)
from src.density_estimator.trainer.helpers import get_dataloader_num_workers
from src.density_estimator.trainer.single_epoch_train import train_one_epoch
from src.density_estimator.utils.reproducibility import seed_worker
from src.density_estimator.utils.visualization import (
    plot_final_losses,
    plot_final_train_metrics,
)


def final_training(
    args: Namespace,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    criterion: torch.nn.Module,
    epochs: int,
    device: torch.device,
    opt_type: str,
    opt_kwargs: dict,
    grad_clip: float | None,
    use_roi_mask: bool,
    patch_size_out: int,
    use_log_counts: bool,
    callback: TrackingCallback,
) -> Dict[str, Any]:
    # Reset model weights
    model.apply(reset_model_weights)
    optimizer = build_optimizer(opt_type, model.parameters(), lr=args.lr, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    # DataLoaders initialization
    final_workers = get_dataloader_num_workers(num_loaders=2)
    logger.info(
        f"Final retraining on full training set … (num_workers={final_workers}/loader)"
    )

    g_final = torch.Generator()
    g_final.manual_seed(args.seed)
    full_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=final_workers > 0,
        num_workers=final_workers,
        worker_init_fn=seed_worker,
        generator=g_final,
    )

    g_test = torch.Generator()
    g_test.manual_seed(args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=final_workers,
        generator=g_test,
    )

    # History initialisation
    train_loss_comp = [f"train_loss_c_{c[0]}" for c in criterion.losses]
    validation_metrics = [f"test_{m}" for m in get_evaluated_metrics_list(criterion)]
    all_evaluated_metrics = ["train_loss"] + train_loss_comp + validation_metrics

    train_hist: Dict[str, List] = {k: [] for k in all_evaluated_metrics}
    # Same structure as "fold_metrics" in CV,
    # or "cv_history" but without the fold dimension (but with test_ instead of val_)

    for epoch in tqdm(range(epochs), desc="Final Train"):
        train_loss, loss_components = train_one_epoch(
            model,
            full_train_loader,
            criterion,
            optimizer,
            device,
            use_roi_mask,
            grad_clip_max_norm=grad_clip,
            patch_size_out=patch_size_out,
            use_log_counts=use_log_counts,
        )

        test_metrics = evaluate_model_on_loader(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_roi_mask=use_roi_mask,
            patch_size_out=patch_size_out,
            use_log_counts=use_log_counts,
        )

        # Step on train_loss to prevent data leakage from test metrics
        scheduler.step(train_loss)

        # Merge metrics into a single dict
        current_epoch_metrics = {}
        current_epoch_metrics["train_loss"] = train_loss
        for c, lc in loss_components.items():
            current_epoch_metrics[f"train_loss_c_{c}"] = lc
        for k in test_metrics.keys():
            key_name = f"test_{k}"
            current_epoch_metrics[key_name] = test_metrics[k]

        # Log metrics to history and on WandB
        for k, v in current_epoch_metrics.items():
            train_hist[k].append(v)

        callback.on_epoch_end(
            phase="final_train",
            epoch=epoch,
            metrics=current_epoch_metrics,
        )

    # ---- Save final-training plots (losses + all metrics) ----
    plot_final_losses(
        train_hist,
        save_path=os.path.join(args.output_dir, "final_train_losses.png"),
    )
    plot_final_train_metrics(
        train_hist,
        save_path=os.path.join(args.output_dir, "final_train_metrics.png"),
    )

    return train_hist
