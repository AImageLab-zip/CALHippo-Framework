import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim_func


class ForegroundL1Loss(nn.Module):
    def __init__(self, fg_weight: float = 10.0, foreground_cutoff: float = 0.1):
        super().__init__()
        self.fg_weight = fg_weight
        self.foreground_cutoff = foreground_cutoff

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """
        # Calculate raw absolute error per pixel
        abs_error = torch.abs(preds - targets)

        # Create a weight map: 1.0 for background, 'fg_weight' for cell regions
        # We use a threshold to define where the Gaussian targets actually are
        weight_map = torch.where(targets > self.foreground_cutoff, self.fg_weight, 1.0)

        # Apply the weight map and compute the mean
        weighted_l1 = torch.mean(abs_error * weight_map)

        return weighted_l1


# FIXME: deprecated. Use the combination of ForegroundL1Loss + SSIMLoss + CountLoss instead
class ForegroundLoss(nn.Module):
    """
    Advanced Density Loss for sparse WSI counting.
    Combines Foreground-Weighted L1, SSIM, and Explicit Count (Integral) Loss.
    """

    def __init__(
        self,
        lambda_ssim: float = 0.5,
        lambda_count: float = 0.1,  # New: Weight for the total count penalty
        fg_weight: float = 10.0,  # New: Multiplier for pixels that contain cells
        foreground_cutoff: float = 0.1,  # New: Threshold to distinguish fg from bg
    ):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_count = lambda_count
        self.fg_weight = fg_weight
        self.foreground_cutoff = foreground_cutoff

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds/targets shape: (B, C, H, W)
        """
        B, C, H, W = preds.shape

        # -----------------------------------------------------------
        # 1. Foreground-Weighted L1 Loss (Fixes the smeared visuals)
        # -----------------------------------------------------------
        # Calculate raw absolute error per pixel
        abs_error = torch.abs(preds - targets)

        # Create a weight map: 1.0 for background, 'fg_weight' for cell regions
        # We use a threshold to define where the Gaussian targets actually are
        weight_map = torch.where(targets > self.foreground_cutoff, self.fg_weight, 1.0)

        # Apply the weight map and compute the mean
        weighted_l1 = torch.mean(abs_error * weight_map)

        # -----------------------------------------------------------
        # 2. Structural Similarity (SSIM)
        # -----------------------------------------------------------
        dynamic_range = (targets.max() - targets.min()).item() + 1e-4
        ssim_val = ssim_func(preds, targets, data_range=dynamic_range)
        ssim_loss = 1.0 - ssim_val

        # -----------------------------------------------------------
        # 3. Explicit Count Loss (Fixes the high NAE / Class Disproportion)
        # -----------------------------------------------------------
        # Sum over spatial dimensions (H, W) to get the total count per image, per channel
        # Shape becomes (B, C)
        pred_counts = preds.sum(dim=(2, 3))
        target_counts = targets.sum(dim=(2, 3))

        # L1 difference of the total counts, averaged across batches and classes
        # This forces the network to preserve the volume of the minority classes!
        count_loss = torch.mean(torch.abs(pred_counts - target_counts))

        # -----------------------------------------------------------
        # Total Loss Calculation
        # -----------------------------------------------------------
        total_loss = (
            weighted_l1
            + (self.lambda_ssim * ssim_loss)
            + (self.lambda_count * count_loss)
        )

        return total_loss
