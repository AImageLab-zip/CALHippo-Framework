import torch
import torch.nn as nn
import torch.nn.functional as F


class GridCountLoss(nn.Module):
    def __init__(
        self,
        grid_size: int = 4,  # Splits 128x128 into a 4x4 grid of 32x32 patches
        lambda_grid: float = 0.5,  # Weight for the local grid-count penalty
        fg_weight: float = 1.0,  # Pixel-wise foreground weight
        foreground_cutoff: float = 0.1,  # Threshold to distinguish fg from bg
    ):
        super().__init__()
        self.grid_size = grid_size
        self.lambda_grid = lambda_grid
        self.fg_weight = fg_weight
        self.foreground_cutoff = foreground_cutoff

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds/targets shape: (B, C, 128, 128)
        """
        B, C, H, W = preds.shape

        # -----------------------------------------------------------
        # 1. Foreground-Weighted Pixel L1 (Keeps the shapes tight)
        # -----------------------------------------------------------
        abs_error = torch.abs(preds - targets)
        weight_map = torch.where(
            targets > self.foreground_cutoff, self.fg_weight, 1.0
        )  # Use your verified cutoff
        pixel_loss = torch.mean(abs_error * weight_map)

        # -----------------------------------------------------------
        # 2. Local Grid-Count Loss
        # -----------------------------------------------------------
        # We use AdaptiveAvgPool to shrink the 128x128 map into a 4x4 map.
        # To convert the "average" to a "sum", we multiply by the number of pixels in the sub-grid.
        pixels_per_grid = (H // self.grid_size) * (W // self.grid_size)

        # Shape becomes (B, C, 4, 4) -> Each pixel represents the total cell count in that 32x32 area
        pred_grid_sums = (
            F.adaptive_avg_pool2d(preds, (self.grid_size, self.grid_size))
            * pixels_per_grid
        )
        target_grid_sums = (
            F.adaptive_avg_pool2d(targets, (self.grid_size, self.grid_size))
            * pixels_per_grid
        )

        # Calculate L1 error on these 16 local counts
        grid_count_loss = torch.mean(torch.abs(pred_grid_sums - target_grid_sums))

        # -----------------------------------------------------------
        # 3. Total Loss
        # -----------------------------------------------------------
        # We can drop SSIM because the Grid Loss naturally enforces structure
        total_loss = pixel_loss + (self.lambda_grid * grid_count_loss)

        return total_loss
