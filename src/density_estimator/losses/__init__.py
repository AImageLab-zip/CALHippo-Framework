"""
Loss factory for ``src.density_estimator``.

Public API::

    from src.density_estimator.losses import build_loss

    criterion = build_loss("density", gain_factor=10.0)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .count_loss import (
    AsymmetricNormalizedClassL1PixelLoss,
    CountLoss,
    L1PixelLoss,
    NAECountLoss,
    NormalizedClassL1PixelLoss,
    NormalizedL1PixelLoss,
)
from .fft_loss import NormalizedClassFFTLoss, NormalizedClassPhaseInvariantFFTPatchLoss
from .foreground_aware_loss import ForegroundL1Loss
from .game_loss import GameLoss, NormalizedGameLoss
from .ssim_loss import SSIMLoss


def _build_l1_loss(**kwargs: Any) -> nn.Module:
    return nn.L1Loss()


def _build_smooth_l1_loss(**kwargs: Any) -> nn.Module:
    return nn.SmoothL1Loss()


def _build_mse_loss(**kwargs: Any) -> nn.Module:
    accepted = {"reduction"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return nn.MSELoss(**loss_args)


def _build_ssim_loss(**kwargs: Any) -> nn.Module:
    return SSIMLoss()


def _build_count_loss(**kwargs: Any) -> nn.Module:
    return CountLoss()


def _build_nae_count_loss(**kwargs: Any) -> nn.Module:
    return NAECountLoss()


def _build_l1_pixel_loss(**kwargs: Any) -> nn.Module:
    return L1PixelLoss()


def _build_normalized_class_l1_pixel_loss(**kwargs: Any) -> nn.Module:
    return NormalizedClassL1PixelLoss()


def _build_asymmetric_normalized_class_l1_pixel_loss(**kwargs: Any) -> nn.Module:
    accepted = {"class_weights", "undercount_penalty", "eps"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return AsymmetricNormalizedClassL1PixelLoss(**loss_args)


# def _build_naepixel_count_loss(**kwargs: Any) -> nn.Module:
#     accepted = {"lambda_count", "epsilon"}
#     loss_args = {k: v for k, v in kwargs.items() if k in accepted}
#     return NAEPixelCountLoss(**loss_args)


def _build_normalized_l1_pixel_loss(**kwargs: Any) -> nn.Module:
    accepted = {"log_weight", "eps"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return NormalizedL1PixelLoss(**loss_args)


def _build_foreground_l1_loss(**kwargs: Any) -> nn.Module:
    accepted = {"fg_weight", "foreground_cutoff"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return ForegroundL1Loss(**loss_args)


def _build_game_loss(**kwargs: Any) -> nn.Module:
    accepted = {"l_split"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return GameLoss(**loss_args)


def _build_normalized_game_loss(**kwargs: Any) -> nn.Module:
    accepted = {"l_split", "epsilon", "class_weights"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return NormalizedGameLoss(**loss_args)


def _build_normalized_class_fft_loss(**kwargs: Any) -> nn.Module:
    accepted = {"patch_size", "stride", "loss_type", "eps", "eps_norm", "class_weights"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return NormalizedClassFFTLoss(**loss_args)


def _build_normalized_class_phase_invariant_fft_patch_loss(**kwargs: Any) -> nn.Module:
    accepted = {"patch_size", "stride", "loss_type", "eps", "eps_norm", "class_weights"}
    loss_args = {k: v for k, v in kwargs.items() if k in accepted}
    return NormalizedClassPhaseInvariantFFTPatchLoss(**loss_args)


_LOSS_REGISTRY = {
    "l1": _build_l1_loss,
    "smooth_l1": _build_smooth_l1_loss,
    "nae": _build_nae_count_loss,
    "mse": _build_mse_loss,
    "ssim": _build_ssim_loss,
    "l1_pixel": _build_l1_pixel_loss,
    "normalized_l1_pixel": _build_normalized_l1_pixel_loss,
    "normalized_class_l1_pixel": _build_normalized_class_l1_pixel_loss,
    "asymmetric_normalized_class_l1_pixel": _build_asymmetric_normalized_class_l1_pixel_loss,
    "foreground_l1": _build_foreground_l1_loss,
    "count": _build_count_loss,
    # "naepixel": _build_naepixel_count_loss,
    "game": _build_game_loss,
    "normalized_game": _build_normalized_game_loss,
    "normalized_class_fft": _build_normalized_class_fft_loss,
    "normalized_class_phase_invariant_fft_patch": _build_normalized_class_phase_invariant_fft_patch_loss,
}

"""
Example configuration for a combined loss function in a YAML file:
LOSS:
  - type: l1
    weight: 1.0
  - type: mse
    weight: 1.0
  - type: game
    weight: 0.01
    l_split: 2
  - type: ssim
    weight: 1.0
  - type: count
    weight: 0.01
"""


class CombinedLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses
        # Losses is a list of tuples: (loss_name, loss_module, weight)

    def forward(self, preds, targets):
        total_loss = 0.0

        loss_details = {}
        for loss_name, loss_module, weight in self.losses:
            single_loss_val = weight * loss_module(preds, targets)

            loss_details[loss_name] = single_loss_val.item()
            total_loss += single_loss_val

        return total_loss, loss_details


def build_loss(loss_config: list[dict]) -> nn.Module:
    """
    Build a loss function based on the provided configuration.

    Args:
        loss_config: A list of dictionaries, each containing:
            - 'type': The type of loss (e.g., 'mse', 'ssim', 'foreground_l1', 'count', 'game').
            - 'weight': The weight for this loss component in the total loss.
            - Additional custom arguments specific to the loss type.

    Returns:
        A PyTorch nn.Module representing the combined loss function.
    """

    loss_modules = []

    for config in loss_config:
        loss_type = config.get("type")
        weight = config.get("weight", 1.0)
        if loss_type not in _LOSS_REGISTRY:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        builder = _LOSS_REGISTRY[loss_type]
        loss_module = builder(**config)
        loss_modules.append((loss_type, loss_module, weight))

    return CombinedLoss(loss_modules)
