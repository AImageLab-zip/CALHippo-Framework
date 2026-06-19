from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GameLoss(nn.Module):
    def __init__(self, l_split: int = 2):
        super().__init__()
        self.l_split = l_split

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted density maps ``(B, C, H, W)``.
            targets: Ground-truth density maps ``(B, C, H, W)``.
        """

        # In order to compute the GameLoss, we have to split both H and W in 2^L subregions
        # Then compute the absolute count error in each subregion and sum across all subregions in each channel
        # Finally, we average across the batch and channels

        B, C, H, W = preds.shape
        total_error = torch.zeros(B, C, device=preds.device, dtype=preds.dtype)

        num_dim_patches = 2**self.l_split
        cell_h = H // num_dim_patches
        cell_w = W // num_dim_patches

        for i in range(num_dim_patches):
            for j in range(num_dim_patches):
                pred_cell = preds[
                    :, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]
                target_cell = targets[
                    :, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]

                pred_count = pred_cell.sum(dim=(2, 3))
                target_count = target_cell.sum(dim=(2, 3))

                total_error += torch.abs(pred_count - target_count)

        game_loss = total_error.mean()
        return game_loss


class NormalizedGameLoss(nn.Module):
    def __init__(
        self,
        l_split: int = 2,
        epsilon: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        """
        class_weights: List of multipliers for each class. e.g., [1.0, 1.0, 3.0]
                       to triple the penalty on the 3rd class (Astrocytes).
        """
        super().__init__()
        self.l_split = l_split
        self.epsilon = epsilon
        self.class_weights = class_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = preds.shape

        # 1. Global Count (The Normalizer)
        global_target_count = targets.sum(dim=(2, 3))

        # 2. Fast Grid Summation (Vectorized)
        grid_size = 2**self.l_split
        patch_area = (H // grid_size) * (W // grid_size)

        pred_grids = F.adaptive_avg_pool2d(preds, (grid_size, grid_size)) * patch_area
        target_grids = (
            F.adaptive_avg_pool2d(targets, (grid_size, grid_size)) * patch_area
        )

        # 3. GAME Calculation
        grid_errors = torch.abs(pred_grids - target_grids)
        total_spatial_error = grid_errors.sum(dim=(2, 3))

        # 4. Class-Balanced Normalization -> Shape: (B, C)
        normalized_game = total_spatial_error / (global_target_count + self.epsilon)

        # -----------------------------------------------------------
        # 5. Apply Explicit Class Weights
        # -----------------------------------------------------------
        if self.class_weights is not None:
            # Convert python list to a tensor on the same device/dtype as the predictions
            weights = torch.tensor(
                self.class_weights, device=preds.device, dtype=preds.dtype
            )

            # Reshape to (1, C) so it broadcasts perfectly across the batch (B, C)
            normalized_game = normalized_game * weights.view(1, C)

        # Average across the batch and the 3 channels
        return normalized_game.mean()
