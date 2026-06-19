import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

# FIXME: deprecated. Use the combination of ForegroundL1Loss + SSIMLoss instead


class DensityLoss(nn.Module):
    """
    Combined L1 + SSIM loss for density-map regression.

    ``Total = gain_factor × (L1 + λ_ssim × (1 − SSIM))``

    Args:
        lambda_ssim: Weight for the SSIM term. (Recommended starting point: 0.1 - 0.5)
        gain_factor: Scalar multiplier applied to the total loss to stabilize
            gradient magnitudes when density values are small.
    """

    def __init__(
        self,
        lambda_ssim: float = 0.5,
        gain_factor: float = 1.0,
    ):
        super().__init__()
        # L1 preserves the absolute integral (total cell count) vastly better than MSE
        self.l1 = nn.L1Loss()
        self.lambda_ssim = lambda_ssim
        self.gain_factor = gain_factor

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """
        # 1. Pixel-wise absolute error
        l1_loss = self.l1(preds, targets)

        # 2. Dynamic Data Range for SSIM
        # Prevents math errors if overlapping Gaussians cause target values > 1.0
        # Add a tiny epsilon (1e-4) to prevent division by zero on completely empty patches
        dynamic_range = (targets.max() - targets.min()).item() + 1e-4

        # 3. Structural Similarity
        ssim_val = ssim_func(preds, targets, data_range=dynamic_range)
        ssim_loss = 1.0 - ssim_val

        # 4. Total calculation
        total_loss = l1_loss + (self.lambda_ssim * ssim_loss)

        return total_loss * self.gain_factor
