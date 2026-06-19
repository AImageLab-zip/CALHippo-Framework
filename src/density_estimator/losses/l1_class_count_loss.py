import torch
import torch.nn as nn


class L1ClassCountLoss(nn.Module):
    def __init__(self, lambda_count: float = 5.0, epsilon: float = 1.0):
        """
        epsilon: Added to the denominator of the NAE calculation to prevent
                 division by zero or exploding gradients on empty patches.
        """
        super().__init__()
        self.lambda_count = lambda_count
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets shape: (B, C, H, W)

        # -----------------------------------------------------------
        # 1. Per-Channel Pixel Loss (L1)
        # -----------------------------------------------------------
        # Calculate L1 for each channel independently across the batch and spatial dims
        # Shape becomes (C,)
        pixel_loss_per_channel = torch.abs(preds - targets).mean(dim=(0, 2, 3))

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
        # Now a 20% error in Astrocytes equals a 20% error in Pyramidal cells
        count_loss = nae_per_channel.mean()

        # -----------------------------------------------------------
        # Total Loss
        # -----------------------------------------------------------
        return pixel_loss + (self.lambda_count * count_loss)
