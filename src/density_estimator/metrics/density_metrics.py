"""Evaluation metrics for density estimation: count-based, map-quality, and GAME."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

# ---------------------------------------------------------------------------
# Count-level metrics
# ---------------------------------------------------------------------------


def compute_count_metrics(
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MAE, RMSE, NAE, SRE over cell counts.

    Args:
        pred_counts: Predicted counts per class ``(N, C)``.
        gt_counts: Ground-truth counts per class ``(N, C)``.
        epsilon: Stability constant for division.

    Returns:
        Tuple ``(mae, rmse, nae, sre)`` — each shape ``(C,)``.
    """
    diff = pred_counts - gt_counts
    abs_diff = torch.abs(diff)
    squared_diff = diff**2

    mae = torch.mean(abs_diff, dim=0)
    rmse = torch.sqrt(torch.mean(squared_diff, dim=0))
    nae = torch.mean(abs_diff / (gt_counts + epsilon), dim=0)
    sre = torch.sqrt(torch.mean(squared_diff / (gt_counts + epsilon), dim=0))

    return mae, rmse, nae, sre


# ---------------------------------------------------------------------------
# Map-quality metrics (PSNR / SSIM)
# ---------------------------------------------------------------------------


def compute_map_metrics(
    preds: torch.Tensor,
    masks: torch.Tensor,
    roi_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample, per-class PSNR and SSIM on density maps.

    Args:
        preds: Predicted density maps ``(B, C, H, W)``.
        masks: Ground-truth density maps ``(B, C, H, W)``.
        roi_mask: Optional ROI mask ``(B, C, H, W)`` or ``(B, 1, H, W)``.
            If provided, metrics are computed only within the ROI.

    Returns:
        ``(batch_psnr, batch_ssim)`` — each shape ``(B, C)``.
    """

    # TODO: if provided, use roi_mask to properly compute the metrics only within the ROI

    if roi_mask is not None:
        # Ensure roi_mask is broadcastable to (B, C, H, W)
        if roi_mask.shape[1] == 1 and masks.shape[1] > 1:
            roi_mask = roi_mask.expand(-1, masks.shape[1], -1, -1)
        elif roi_mask.shape[1] != masks.shape[1]:
            raise ValueError(
                "roi_mask channel dimension must be 1 or match the number of classes in masks"
            )

        # Apply ROI mask to preds and masks
        preds = preds * roi_mask
        masks = masks * roi_mask

    batch_size, num_classes, _, _ = preds.shape

    psnr_scores: List[torch.Tensor] = []
    ssim_scores: List[torch.Tensor] = []

    for i in range(batch_size):
        sample_psnr: List[torch.Tensor] = []
        sample_ssim: List[torch.Tensor] = []

        for c in range(num_classes):
            p = preds[i, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            t = masks[i, c].unsqueeze(0).unsqueeze(0)

            d_range = t.max() - t.min()
            if d_range == 0:
                d_range = torch.tensor(1.0, device=t.device)

            val_psnr = psnr_func(p, t, data_range=d_range.item())
            val_ssim = ssim_func(p, t, data_range=d_range.item())

            sample_psnr.append(val_psnr)
            sample_ssim.append(val_ssim)

        psnr_scores.append(torch.stack(sample_psnr))
        ssim_scores.append(torch.stack(sample_ssim))

    return torch.stack(psnr_scores), torch.stack(ssim_scores)  # (B, C)


# ---------------------------------------------------------------------------
# GAME metric (Grid Average Mean absolute Error)
# ---------------------------------------------------------------------------


class GAMEMetric:
    """
    Grid Average Mean absolute Error at multiple grid levels.

    At level *L* the density map is partitioned into a ``2^L × 2^L`` grid and
    the absolute count error is summed across all grid cells.

    Args:
        levels: List of grid levels to evaluate (default ``[0, 1, 2, 3]``).
    """

    def __init__(self, levels: List[int] | None = None):
        self.levels = levels or [0, 1, 2, 3]

    def compute(
        self, pred_density: torch.Tensor, gt_density: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Args:
            pred_density: ``(B, C, H, W)``
            gt_density: ``(B, C, H, W)``

        Returns:
            Dict mapping level → error tensor ``(B, C)``.
        """
        results: Dict[int, torch.Tensor] = {}
        _, _, H, W = pred_density.shape
        total_area = H * W

        for L in self.levels:
            k = 2**L
            scale = total_area / (k * k)

            p_grid = F.adaptive_avg_pool2d(pred_density, (k, k)) * scale
            g_grid = F.adaptive_avg_pool2d(gt_density, (k, k)) * scale

            batch_errors = torch.abs(p_grid - g_grid).sum(dim=(2, 3))
            results[L] = batch_errors  # (B, C)

        return results
