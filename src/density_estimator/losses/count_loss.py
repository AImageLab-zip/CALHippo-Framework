from typing import Any, List, Optional

import torch
import torch.nn as nn


class CountLoss(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """
        pred_counts = preds.sum(dim=(2, 3))
        target_counts = targets.sum(dim=(2, 3))

        count_loss = torch.mean(torch.abs(pred_counts - target_counts))

        return count_loss


class NAECountLoss(nn.Module):
    def __init__(self, epsilon: float = 1.0):
        """
        epsilon: Added to the denominator of the NAE calculation to prevent
                 division by zero or exploding gradients on empty patches.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """
        pred_counts = preds.sum(dim=(2, 3))
        target_counts = targets.sum(dim=(2, 3))

        nae_loss = torch.mean(
            torch.abs(pred_counts - target_counts) / (target_counts + self.epsilon)
        )

        return nae_loss


class L1PixelLoss(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        # L1 is mandatory for ultra-dense peaks. MSE will destroy your count.
        self.l1 = nn.L1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Pixel Error (Preserves sharp peaks)
        pixel_loss = self.l1(preds, targets)

        return pixel_loss


class NormalizedL1PixelLoss(nn.Module):
    def __init__(self, log_weight: bool = False, eps: float = 1.0):
        super().__init__()
        self.eps = eps
        self.log_weight = log_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate standard L1
        pixel_l1 = torch.abs(preds - targets).sum(dim=(1, 2, 3))
        # shape becomes (B,) instead of (B, C, H, W)

        # Calculate GT sums per image in batch
        gt_sums = targets.sum(
            dim=(1, 2, 3)
        )  # sums over all channels and pixels, per image
        # shape becomes (B,)

        # Normalize: Error relative to the total population
        normalized_loss = pixel_l1 / (gt_sums + self.eps)
        # shape remains (B,)

        return normalized_loss.mean()


class NormalizedClassL1PixelLoss(nn.Module):
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Calculate standard L1 per image, PER CHANNEL
        # Shape becomes (B, C) instead of (B,)
        pixel_l1 = torch.abs(preds - targets).sum(dim=(2, 3))

        # we obtain an l1 distance between the 3 channels per image, inside a batch size of B. So we have a (B, C) tensor where each element is the sum of absolute differences for that channel in that image.

        # 2. Calculate GT sums per image, PER CHANNEL
        # Shape becomes (B, C)
        gt_sums = targets.sum(dim=(2, 3))
        # now the regularization is per class, not per image. So we have a (B, C) tensor where each element is the sum of the ground truth density for that channel in that image.

        # 3. Normalize: Error relative to that specific class's population
        # Shape remains (B, C)
        normalized_loss = pixel_l1 / (gt_sums + self.eps)

        # in this way, a 30% error in the class with 1000 cells will be treated as equally bad as a 30% error in the class with 10 cells, because both will yield a normalized loss of 0.3. This prevents the model from neglecting the smaller classes and encourages it to perform well across all classes, regardless of their absolute counts.

        # 4. Average equally across the batch AND the 3 channels
        return normalized_loss.mean()


class AsymmetricNormalizedClassL1PixelLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1.0,
        undercount_penalty: float = 2.0,
        class_weights: Optional[List[float]] = None,
    ):
        """
        undercount_penalty: Multiplier applied ONLY when Pred < Target.
        class_weights: List of multipliers for each class. e.g., [1.0, 1.0, 3.0]
        """
        super().__init__()
        self.eps = eps
        self.undercount_penalty = undercount_penalty
        self.class_weights = class_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = preds.shape

        # 1. Calculate raw error (Pred - Target)
        diff = preds - targets

        # 2. Asymmetric Penalty
        # If diff is negative (Pred < Target), multiply by the penalty (e.g., 2.0)
        # If diff is positive (Pred > Target), keep the standard absolute error
        asymmetric_error = torch.where(
            diff < 0, torch.abs(diff) * self.undercount_penalty, torch.abs(diff)
        )

        # 3. Sum per channel -> Shape: (B, C)
        pixel_l1 = asymmetric_error.sum(dim=(2, 3))

        # 4. Normalize by GT sums -> Shape: (B, C)
        gt_sums = targets.sum(dim=(2, 3))
        normalized_loss = pixel_l1 / (gt_sums + self.eps)

        # -----------------------------------------------------------
        # 5. Apply Explicit Class Weights
        # -----------------------------------------------------------
        if self.class_weights is not None:
            # Create a tensor of weights on the same device/dtype as the predictions
            weights = torch.tensor(
                self.class_weights, device=preds.device, dtype=preds.dtype
            )

            # Reshape to (1, C) to broadcast seamlessly across the batch shape (B, C)
            normalized_loss = normalized_loss * weights.view(1, C)

        # 6. Average equally across batch and channels
        return normalized_loss.mean()
