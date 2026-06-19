from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.density_estimator.trainer.helpers import patchify


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_roi_mask: bool = False,
    grad_clip_max_norm: float | None = None,
    patch_size_out: int = 1,
    use_log_counts: bool = False,
) -> tuple[float, dict]:
    """Run one training epoch, return mean batch loss and dict of loss components."""

    model.train()
    batch_losses: List[float] = []
    # criterion.losses is a list of (name, module, weight)
    batch_loss_details: dict[str, List[float]] = {c[0]: [] for c in criterion.losses}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        preds = model(images)

        if use_roi_mask and "roi_mask" in batch:
            # Expand roi_mask to match density map channels if needed
            roi_mask = batch["roi_mask"].to(device)
            roi_mask = roi_mask.expand_as(preds).float()

            preds = preds * roi_mask
            masks = masks * roi_mask

        # Patchify (+ optional log-transform) targets when using patched output
        if patch_size_out > 1:
            masks = patchify(masks, patch_size_out, use_log=use_log_counts)

        loss, loss_details = criterion(preds, masks)
        loss.backward()
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_max_norm
            )
        optimizer.step()

        batch_losses.append(loss.item())
        for c in criterion.losses:
            loss_name = c[0]  # c is (name, loss_func, weight)
            batch_loss_details[loss_name].append(loss_details[loss_name])

    avg_loss = float(np.mean(batch_losses))
    avg_loss_details = {
        name: float(np.mean(values)) for name, values in batch_loss_details.items()
    }

    return avg_loss, avg_loss_details
