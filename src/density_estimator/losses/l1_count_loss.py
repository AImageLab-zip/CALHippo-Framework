import torch
import torch.nn as nn


class L1CountLoss(nn.Module):
    def __init__(self, lambda_count: float = 0.1):
        super().__init__()
        # L1 is mandatory for ultra-dense peaks. MSE will destroy your count.
        self.l1 = nn.L1Loss()
        self.lambda_count = lambda_count

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Pixel Error (Preserves sharp peaks)
        pixel_loss = self.l1(preds, targets)

        pred_counts = preds.sum(dim=(2, 3))
        target_counts = targets.sum(dim=(2, 3))

        # Divide by target counts (+ epsilon) to turn the error into a scale of ~0.0 to ~1.0
        count_loss = torch.mean(
            torch.abs(pred_counts - target_counts) / (target_counts + 1e-8)
        )

        return pixel_loss + (self.lambda_count * count_loss)
