import torch
import torch.nn as nn
import torch.nn.functional as F


class L1ClassGAMELoss(nn.Module):
    def __init__(
        self,
        lambda_count: float = 5.0,
        epsilon: float = 1.0,
        fg_weight: float = 10.0,
        threshold: float = 0.1,
        game_levels: list = [
            0,
            1,
            2,
            3,
        ],  # Computes GAME across 1x1, 2x2, 4x4, and 8x8 grids
    ):
        """
        epsilon: Added to denominator to prevent division by zero.
        fg_weight: Multiplier for pixels where target > threshold.
        game_levels: The powers of 2 for grid splitting.
        """
        super().__init__()
        self.lambda_count = lambda_count
        self.epsilon = epsilon
        self.fg_weight = fg_weight
        self.threshold = threshold
        self.game_levels = game_levels

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets shape: (B, C, H, W)
        B, C, H, W = preds.shape

        # -----------------------------------------------------------
        # 1. Foreground-Weighted Pixel Loss (L1)
        # -----------------------------------------------------------
        weight_map = torch.where(targets > self.threshold, self.fg_weight, 1.0)
        weighted_error = torch.abs(preds - targets) * weight_map

        # Average across spatial and batch dims -> Shape: (C,)
        pixel_loss_per_channel = weighted_error.mean(dim=(0, 2, 3))
        pixel_loss = pixel_loss_per_channel.mean()

        # -----------------------------------------------------------
        # 2. Multi-Scale Class-Balanced GAME Loss
        # -----------------------------------------------------------
        # To maintain CLASS BALANCE, we must scale the spatial grid errors
        # by the GLOBAL target count of that specific class.
        global_target_counts = targets.sum(
            dim=(2, 3), keepdim=True
        )  # Shape: (B, C, 1, 1)

        total_game_loss = 0.0

        for level in self.game_levels:
            # Calculate grid divisions: Level 0->1x1, Level 1->2x2, Level 2->4x4, Level 3->8x8
            grid_size = 2**level

            # Fast grid summation
            patch_area = (H // grid_size) * (W // grid_size)
            pred_grids = (
                F.adaptive_avg_pool2d(preds, (grid_size, grid_size)) * patch_area
            )
            target_grids = (
                F.adaptive_avg_pool2d(targets, (grid_size, grid_size)) * patch_area
            )

            # 1. Calculate absolute error inside each grid
            grid_abs_error = torch.abs(pred_grids - target_grids)

            # 2. Sum the errors across all grids for this image/channel -> Shape: (B, C, 1, 1)
            level_abs_error = grid_abs_error.sum(dim=(2, 3), keepdim=True)

            # 3. CLASS BALANCE: Divide by the global target count to convert to a percentage
            level_nae_error = level_abs_error / (global_target_counts + self.epsilon)

            # Average across the batch -> Shape: (C, 1, 1)
            level_nae_per_channel = level_nae_error.mean(dim=0)

            # Average across the 3 channels and add to the running total
            total_game_loss += level_nae_per_channel.mean()

        # Average the loss across the 4 GAME levels
        mean_game_loss = total_game_loss / len(self.game_levels)

        # -----------------------------------------------------------
        # Total Loss
        # -----------------------------------------------------------
        return pixel_loss + (self.lambda_count * mean_game_loss)
