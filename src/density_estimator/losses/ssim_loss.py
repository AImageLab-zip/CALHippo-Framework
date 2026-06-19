import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim_func


class SSIMLoss(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """

        # Dynamic Data Range for SSIM
        # Prevents math errors if overlapping Gaussians cause target values > 1.0
        dynamic_range = (targets.max() - targets.min()).item() + 1e-4

        ssim_val = ssim_func(preds, targets, data_range=dynamic_range)
        ssim_loss = 1.0 - ssim_val

        return ssim_loss
