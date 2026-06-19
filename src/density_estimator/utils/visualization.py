"""Visualization utilities for density estimation predictions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.density_estimator.metrics.density_metrics import compute_map_metrics

# Non-interactive backend — plots are saved, never displayed.
plt.switch_backend("Agg")


# ---------------------------------------------------------------------------
# Patched-output helpers
# ---------------------------------------------------------------------------


def _upsample_patched_pred(
    pred_tensor: torch.Tensor,
    target_hw: tuple[int, int],
    patch_size_out: int,
    use_log_counts: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Convert patched predictions to full-resolution density + counts.

    When *use_log_counts* is ``True``, applies ``expm1`` to invert the
    log1p compression.  Otherwise predictions are treated as raw counts.

    Returns:
        mask_pred: ``(C, H, W)`` numpy array suitable for overlay.
        pred_counts: per-class total counts.
    """
    if use_log_counts:
        pred_counts_t = torch.clamp(torch.expm1(pred_tensor), min=0.0)  # (1, C, h, w)
    else:
        pred_counts_t = torch.clamp(pred_tensor, min=0.0)
    per_class_counts = [
        float(pred_counts_t[0, c].sum()) for c in range(pred_counts_t.shape[1])
    ]
    # Upsample to full res and rescale to per-pixel density
    up = F.interpolate(pred_counts_t, size=target_hw, mode="nearest")
    mask_pred = (up[0] / (patch_size_out**2)).cpu().numpy()  # (C, H, W)
    return mask_pred, per_class_counts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Save *fig* with a thin black border frame, then close it."""
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(0.8)
    fig.savefig(
        str(path),
        dpi=dpi,
        bbox_inches="tight",
        edgecolor=fig.get_edgecolor(),
        facecolor=fig.get_facecolor(),
        pad_inches=0.05,
    )
    plt.close(fig)


def _create_overlay(
    base_image: np.ndarray,
    density_map: np.ndarray,
    colors: list[str],
    gain: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-blend density channels onto an RGB base image.

    Args:
        base_image: ``(H, W, 3)`` float image in ``[0, 1]``.
        density_map: ``(C, H, W)`` density maps.
        colors: One matplotlib colour name per channel.
        gain: Blending intensity multiplier.
        roi_mask: Optional ``(H, W)`` binary mask. If provided, draws a
            semi-transparent green overlay and yellow contour for ROI boundary.

    Returns:
        Composited ``(H, W, 3)`` float image clipped to ``[0, 1]``.
    """
    composite = base_image.copy()
    for c_idx, color_name in enumerate(colors):
        if c_idx >= density_map.shape[0]:
            break
        channel = density_map[c_idx]
        if channel.max() <= 0:
            continue
        alpha = np.clip(channel * (gain / 255.0), 0, 0.8)
        rgb = np.array(mcolors.to_rgb(color_name))
        alpha_3d = alpha[:, :, np.newaxis]
        color_layer = np.ones_like(composite) * rgb
        composite = composite * (1 - alpha_3d) + color_layer * alpha_3d

    # Overlay ROI mask if provided (border only)
    if roi_mask is not None:
        roi_mask_2d = (
            roi_mask.astype(bool) if roi_mask.ndim == 2 else roi_mask[0].astype(bool)
        )

        # Draw yellow contour around ROI boundary
        roi_uint8 = roi_mask_2d.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_mask = np.zeros_like(roi_uint8)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)
        contour_bool = contour_mask > 0
        composite[contour_bool] = np.array([1.0, 1.0, 0.0])  # Yellow contour

    return np.clip(composite, 0, 1)


def _denormalise_image(
    img_tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
    device: str | torch.device,
) -> np.ndarray:
    """Convert a ``(3, H, W)`` normalised tensor to ``(H, W, 3)`` in ``[0, 1]``."""
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    img = torch.clamp(img_tensor * std_t + mean_t, 0, 1)
    return img.cpu().permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Cross-validation loss plot (train + val)
# ---------------------------------------------------------------------------


def _pad_ragged_folds(fold_lists: list) -> np.ndarray:
    """Pad ragged per-fold epoch lists to equal length (forward-fill).

    When early stopping triggers at different epochs per fold, sublists
    have different lengths.  Shorter folds are padded by repeating their
    **last** value so ``np.array`` produces a rectangular array.
    """
    max_len = max(len(f) for f in fold_lists)
    padded = []
    for fold in fold_lists:
        shortage = max_len - len(fold)
        if shortage > 0:
            fold = list(fold) + [fold[-1]] * shortage
        padded.append(fold)
    return np.array(padded)


def plot_cv_losses(
    cv_history: Dict[str, list],
    save_path: str | Path,
) -> None:
    """Plot train and val loss over epochs (mean ± std across folds)."""
    num_epochs = max(len(f) for f in cv_history["train_loss"])
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, colour, label in [
        ("train_loss", "tab:blue", "Train Loss"),
        ("val_loss", "tab:orange", "Val Loss"),
    ]:
        data = _pad_ragged_folds(cv_history[key])
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(epochs, mean, label=label, color=colour)
        ax.fill_between(epochs, mean - std, mean + std, color=colour, alpha=0.2)

    ax.set_title("Cross-Validation Losses (mean ± std)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    logger.info(f"CV loss plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Cross-validation metric plots (one subplot per metric)
# ---------------------------------------------------------------------------

_CV_METRIC_PLOT_DEFS = [
    ("val_mae", "MAE"),
    ("val_rmse", "RMSE"),
    ("val_nae", "NAE"),
    ("val_sre", "SRE"),
    ("val_psnr", "PSNR"),
    ("val_ssim", "SSIM"),
]


def plot_cv_metrics(
    cv_history: Dict[str, list],
    save_path: str | Path,
) -> None:
    """Plot all validation metrics over epochs (mean ± std)."""
    present = [(k, t) for k, t in _CV_METRIC_PLOT_DEFS if k in cv_history]
    n = len(present)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (key, title) in enumerate(present):
        ax = axes_flat[i]
        data = _pad_ragged_folds(cv_history[key])
        if data.ndim == 3:
            data = data.mean(axis=2)
        num_epochs = data.shape[1]
        epochs = np.arange(1, num_epochs + 1)

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(epochs, mean, label=f"Val {title}", color="tab:green")
        ax.fill_between(epochs, mean - std, mean + std, color="tab:green", alpha=0.2)
        ax.set_title(f"CV — {title}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    logger.info(f"CV metrics plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Final-training loss curves (train + holdout test)
# ---------------------------------------------------------------------------


def plot_final_losses(
    final_train_history: Dict[str, list],
    save_path: str | Path,
) -> None:
    """Plot per-epoch train loss and hold-out test loss during final retraining."""
    num_epochs = len(final_train_history["train_loss"])
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs,
        final_train_history["train_loss"],
        label="Train Loss",
        color="tab:blue",
    )
    ax.plot(
        epochs,
        final_train_history["test_loss"],
        label="Test Loss",
        color="tab:red",
    )
    ax.set_title("Final Retraining — Train vs. Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    logger.info(f"Final loss plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Final-training metric curves (test set, mirrors CV metric plots)
# ---------------------------------------------------------------------------

_FINAL_METRIC_PLOT_DEFS = [
    ("test_mae", "MAE"),
    ("test_rmse", "RMSE"),
    ("test_nae", "NAE"),
    ("test_sre", "SRE"),
    ("test_psnr", "PSNR"),
    ("test_ssim", "SSIM"),
]


def plot_final_train_metrics(
    final_train_history: Dict[str, list],
    save_path: str | Path,
) -> None:
    """Plot all test-set metrics over epochs during final retraining.

    Each per-class metric (stored as a list of ``np.ndarray(C,)`` values)
    is averaged across classes and plotted as a single line — analogous
    to :func:`plot_cv_metrics` but for a single training run (no fold
    envelope).
    """
    present = [(k, t) for k, t in _FINAL_METRIC_PLOT_DEFS if k in final_train_history]
    n = len(present)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (key, title) in enumerate(present):
        ax = axes_flat[i]
        data = np.array(final_train_history[key])  # (E,) or (E, C)
        if data.ndim == 2:
            # Per-class → mean across classes
            data = data.mean(axis=1)
        num_epochs = len(data)
        epochs = np.arange(1, num_epochs + 1)

        ax.plot(epochs, data, label=f"Test {title}", color="tab:green")
        ax.set_title(f"Final Train — {title}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    logger.info(f"Final train metrics plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Prediction summary: 3-column (Input | GT overlay | Pred overlay)
# ---------------------------------------------------------------------------


def plot_prediction_summary(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str | torch.device,
    save_path: str | Path,
    num_samples: int = 4,
    class_names: List[str] | None = None,
    class_colors: List[str] | None = None,
    mean_list: List[float] | None = None,
    std_list: List[float] | None = None,
    gain: float = 150.0,
    seed: int = 42,
    show_roi_mask: bool = False,
    patch_size_out: int = 1,
    use_log_counts: bool = False,
) -> None:
    """3-column figure: Input | GT overlay (all classes) | Pred overlay.

    One row per sample.  Titles show total and per-class counts,
    overall percentage error, mean PSNR and SSIM.
    Saved to *save_path*; nothing is displayed.
    """
    class_names = class_names or ["Pyramidal", "Interneuron", "Astrocyte"]
    class_colors = class_colors or ["red", "cyan", "blue"]
    mean_list = mean_list or [0.7637, 0.7637, 0.7637]
    std_list = std_list or [0.0703, 0.0703, 0.0703]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    n_rows = len(indices)

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            mask_gt = sample["mask"].numpy()  # (C, H, W)

            # Get ROI mask if available and requested
            roi_mask_np = None
            if show_roi_mask and "roi_mask" in sample:
                roi_mask_np = sample["roi_mask"].numpy()
                if roi_mask_np.ndim == 3:
                    roi_mask_np = roi_mask_np[0]  # Take first channel if multi-channel

            # Convert ROI mask to tensor for metrics computation if available
            roi_mask_tensor = None
            if roi_mask_np is not None:
                roi_mask_tensor = (
                    torch.from_numpy(roi_mask_np).unsqueeze(0).unsqueeze(0).to(device)
                )

            output = model(img_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred_tensor = torch.relu(output)

            # --- Patched-output handling ---
            if patch_size_out > 1:
                target_hw = mask_gt.shape[-2:]
                mask_pred, pred_counts = _upsample_patched_pred(
                    pred_tensor,
                    target_hw,
                    patch_size_out,
                    use_log_counts=use_log_counts,
                )
                # PSNR / SSIM in the model's output space
                gt_for_metrics = F.avg_pool2d(
                    sample["mask"].unsqueeze(0).to(device),
                    kernel_size=patch_size_out,
                ) * (patch_size_out**2)
                if use_log_counts:
                    gt_for_metrics = torch.log1p(gt_for_metrics)
                sample_psnr, sample_ssim = compute_map_metrics(
                    pred_tensor,
                    gt_for_metrics,
                    roi_mask=roi_mask_tensor,
                )
            else:
                mask_pred = pred_tensor.cpu().numpy()[0]  # (C, H, W)
                pred_counts = [
                    float(mask_pred[c].sum()) for c in range(len(class_names))
                ]
                sample_psnr, sample_ssim = compute_map_metrics(
                    pred_tensor,
                    sample["mask"].unsqueeze(0).to(device),
                    roi_mask=roi_mask_tensor,
                )
            mean_psnr = float(sample_psnr.mean())
            mean_ssim = float(sample_ssim.mean())

            img_rgb = _denormalise_image(img_tensor[0], mean_list, std_list, device)

            # Overlays with all classes blended together (with ROI mask if available)
            overlay_gt = _create_overlay(
                img_rgb, mask_gt, class_colors, gain, roi_mask_np
            )
            overlay_pred = _create_overlay(
                img_rgb, mask_pred, class_colors, gain, roi_mask_np
            )

            # Per-class counts (filtered by ROI mask if provided)
            if roi_mask_np is not None:
                roi_bool = roi_mask_np.astype(bool)
                gt_counts = [
                    float(mask_gt[c][roi_bool].sum()) for c in range(len(class_names))
                ]
                pred_counts = [
                    float(mask_pred[c][roi_bool].sum()) for c in range(len(class_names))
                ]
            else:
                gt_counts = [float(mask_gt[c].sum()) for c in range(len(class_names))]
                pred_counts = [
                    float(mask_pred[c].sum()) for c in range(len(class_names))
                ]
            total_gt = sum(gt_counts)
            total_pred = sum(pred_counts)

            # Overall percentage error
            pct_err = (total_pred - total_gt) / (total_gt + 1e-5) * 100.0
            err_color = "darkgreen" if abs(pct_err) < 1.0 else "darkred"

            gt_parts = ", ".join(
                f"{n}: {v:.1f}" for n, v in zip(class_names, gt_counts)
            )
            pred_parts = ", ".join(
                f"{n}: {v:.1f}" for n, v in zip(class_names, pred_counts)
            )

            # Col 0 — Input
            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(
                f"Sample {idx} — Input", fontsize=11, fontweight="bold"
            )
            axes[row, 0].axis("off")

            # Col 1 — GT overlay
            axes[row, 1].imshow(overlay_gt)
            axes[row, 1].set_title(
                f"GT  total: {total_gt:.1f}\n{gt_parts}",
                fontsize=10,
                fontweight="bold",
            )
            axes[row, 1].axis("off")

            # Col 2 — Pred overlay
            axes[row, 2].imshow(overlay_pred)
            axes[row, 2].set_title(
                f"Pred total: {total_pred:.1f} "
                f"(Err: {pct_err:+.1f}%)\n"
                f"{pred_parts}\n"
                f"PSNR: {mean_psnr:.1f} dB | SSIM: {mean_ssim:.3f}",
                fontsize=10,
                fontweight="bold",
                color=err_color,
            )
            axes[row, 2].axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    logger.info(f"Prediction summary plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Per-class prediction overlay (2 rows per sample × num_classes columns)
# ---------------------------------------------------------------------------


def plot_prediction_per_class(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str | torch.device,
    save_path: str | Path,
    num_samples: int = 4,
    class_names: List[str] | None = None,
    class_colors: List[str] | None = None,
    mean_list: List[float] | None = None,
    std_list: List[float] | None = None,
    gain: float = 150.0,
    seed: int = 42,
    show_roi_mask: bool = False,
    patch_size_out: int = 1,
    use_log_counts: bool = False,
) -> None:
    """Per-class overlay: 2 rows per sample (GT / Pred) × one column per class.

    Each cell shows the single-class density overlay on the input image.
    Titles include per-class counts, percentage error, PSNR and SSIM.
    Saved to *save_path*; nothing is displayed.
    """
    class_names = class_names or ["Pyramidal", "Interneuron", "Astrocyte"]
    class_colors = class_colors or ["red", "cyan", "blue"]
    mean_list = mean_list or [0.7637, 0.7637, 0.7637]
    std_list = std_list or [0.0703, 0.0703, 0.0703]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    num_classes = len(class_names)
    total_rows = len(indices) * 2

    fig, axes = plt.subplots(
        total_rows, num_classes, figsize=(6 * num_classes, 5 * total_rows)
    )
    # Ensure axes is always 2-D (rows, cols) — handles num_classes=1 or total_rows=1
    axes = np.atleast_2d(axes).reshape(total_rows, num_classes)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            mask_gt = sample["mask"].numpy()

            # Get ROI mask if available and requested
            roi_mask_np = None
            if show_roi_mask and "roi_mask" in sample:
                roi_mask_np = sample["roi_mask"].numpy()
                if roi_mask_np.ndim == 3:
                    roi_mask_np = roi_mask_np[0]  # Take first channel if multi-channel

            # Convert ROI mask to tensor for metrics computation if available
            roi_mask_tensor = None
            if roi_mask_np is not None:
                roi_mask_tensor = (
                    torch.from_numpy(roi_mask_np).unsqueeze(0).unsqueeze(0).to(device)
                )

            output = model(img_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred_tensor = torch.relu(output)

            # --- Patched-output handling ---
            if patch_size_out > 1:
                target_hw = mask_gt.shape[-2:]
                mask_pred, _ = _upsample_patched_pred(
                    pred_tensor,
                    target_hw,
                    patch_size_out,
                    use_log_counts=use_log_counts,
                )
                gt_for_metrics = F.avg_pool2d(
                    sample["mask"].unsqueeze(0).to(device),
                    kernel_size=patch_size_out,
                ) * (patch_size_out**2)
                if use_log_counts:
                    gt_for_metrics = torch.log1p(gt_for_metrics)
                sample_psnr, sample_ssim = compute_map_metrics(
                    pred_tensor,
                    gt_for_metrics,
                    roi_mask=roi_mask_tensor,
                )
                # Per-class counts from model output
                if use_log_counts:
                    pred_counts_all = torch.clamp(torch.expm1(pred_tensor), min=0.0)
                else:
                    pred_counts_all = torch.clamp(pred_tensor, min=0.0)
                per_class_pred_counts = [
                    float(pred_counts_all[0, c].sum()) for c in range(num_classes)
                ]
            else:
                mask_pred = pred_tensor.cpu().numpy()[0]
                sample_psnr, sample_ssim = compute_map_metrics(
                    pred_tensor,
                    sample["mask"].unsqueeze(0).to(device),
                    roi_mask=roi_mask_tensor,
                )
                per_class_pred_counts = [
                    float(mask_pred[c].sum()) for c in range(num_classes)
                ]

            cls_psnr = sample_psnr[0].cpu().numpy()  # (C,)
            cls_ssim = sample_ssim[0].cpu().numpy()  # (C,)

            img_rgb = _denormalise_image(img_tensor[0], mean_list, std_list, device)

            row_gt = i * 2
            row_pred = i * 2 + 1

            for c in range(num_classes):
                color = class_colors[c]
                name = class_names[c]

                # GT overlay (single class, with ROI mask if available)
                # Filter counts by ROI mask if provided
                if roi_mask_np is not None:
                    roi_bool = roi_mask_np.astype(bool)
                    count_gt = float(mask_gt[c][roi_bool].sum())
                    count_pred = float(mask_pred[c][roi_bool].sum())
                else:
                    count_gt = float(mask_gt[c].sum())
                    count_pred = float(mask_pred[c].sum())

                overlay_gt = _create_overlay(
                    img_rgb, mask_gt[c][np.newaxis], [color], gain, roi_mask_np
                )

                # Pred overlay (single class, with ROI mask if available)
                overlay_pred = _create_overlay(
                    img_rgb, mask_pred[c][np.newaxis], [color], gain, roi_mask_np
                )

                # Percentage error
                safe_gt = count_gt if count_gt > 0 else 1e-5
                pct_err = ((count_pred - count_gt) / safe_gt) * 100.0
                err_color = "darkgreen" if abs(pct_err) < 5.0 else "darkred"

                # GT row
                ax_gt = axes[row_gt, c]
                ax_gt.imshow(overlay_gt)
                if c == 0:
                    ax_gt.set_ylabel(
                        f"Sample {idx}\nGROUND TRUTH",
                        fontsize=12,
                        fontweight="bold",
                    )
                ax_gt.set_title(
                    f"{name}\nCount: {count_gt:.1f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_gt.axis("off")

                # Pred row
                ax_pred = axes[row_pred, c]
                ax_pred.imshow(overlay_pred)
                if c == 0:
                    ax_pred.set_ylabel("PREDICTION", fontsize=12, fontweight="bold")
                ax_pred.set_title(
                    f"{name}\nPred: {count_pred:.1f} ({pct_err:+.1f}%)\n"
                    f"PSNR: {cls_psnr[c]:.1f} dB | SSIM: {cls_ssim[c]:.3f}",
                    fontsize=11,
                    fontweight="bold",
                    color=err_color,
                )
                ax_pred.axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    logger.info(f"Per-class prediction plot saved → {save_path}")
