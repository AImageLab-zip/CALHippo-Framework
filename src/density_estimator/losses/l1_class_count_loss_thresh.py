import torch
import torch.nn as nn


class L1ClassCountTreshLoss(nn.Module):
    def __init__(
        self,
        lambda_count: float = 5.0,
        epsilon: float = 1.0,
        fg_weight: float = 10.0,
        threshold: float = 0.1,
    ):
        """
        epsilon: Added to the denominator of the NAE calculation.
        fg_weight: Multiplier applied to the pixel loss where target > threshold.
        threshold: The density value that defines the foreground peaks.
        """
        super().__init__()
        self.lambda_count = lambda_count
        self.epsilon = epsilon
        self.fg_weight = fg_weight
        self.threshold = threshold

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets shape: (B, C, H, W)

        # -----------------------------------------------------------
        # 1. Foreground-Weighted Per-Channel Pixel Loss (L1)
        # -----------------------------------------------------------
        # Create weight map: multiply by fg_weight if target > threshold, else 1.0
        weight_map = torch.where(targets > self.threshold, self.fg_weight, 1.0)

        # Calculate raw absolute error, then apply the weight map
        abs_error = torch.abs(preds - targets)
        weighted_error = abs_error * weight_map

        # Calculate L1 for each channel independently across the batch and spatial dims
        # Shape becomes (C,)
        pixel_loss_per_channel = weighted_error.mean(dim=(0, 2, 3))

        # Average the 3 channels equally -> Shape (1,)
        pixel_loss = pixel_loss_per_channel.mean()

        # -----------------------------------------------------------
        # 2. Per-Channel Relative Count Loss (NAE)
        # -----------------------------------------------------------
        # Sum spatial dimensions to get counts -> Shape (B, C)
        pred_counts = preds.sum(dim=(2, 3))
        target_counts = targets.sum(dim=(2, 3))

        # Calculate NAE per image, per channel
        # Using abs(Pred - Target) / (Target + Epsilon)
        batch_channel_nae = torch.abs(pred_counts - target_counts) / (
            target_counts + self.epsilon
        )

        # Average across the batch -> Shape (C,)
        nae_per_channel = batch_channel_nae.mean(dim=0)

        # Average the 3 channels equally -> Shape (1,)
        count_loss = nae_per_channel.mean()

        # -----------------------------------------------------------
        # Total Loss
        # -----------------------------------------------------------
        return pixel_loss + (self.lambda_count * count_loss)
