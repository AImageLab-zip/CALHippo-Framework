"""
Model factory for ``src.density_estimator``.

Public API::

    from src.density_estimator.models import build_model

    model = build_model("plain_conv_unet", input_channels=3, num_classes=3)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .csrnet_adapter import build_csrnet_adapter
from .mcnn_adapter import build_mcnn_adapter
from .model_helpers import reset_model_weights
from .unet import (
    build_flexible_unet_relu,
    build_plain_conv_unet,
    build_plain_conv_unet_bilinear_grouped,
    build_plain_conv_unet_relu,
    build_residual_encoder_unet,
    build_resnet_enc,
)

# Registry: model_type  →  builder callable
_MODEL_REGISTRY = {
    "plain_conv_unet": build_plain_conv_unet,
    "plain_conv_unet_relu": build_plain_conv_unet_relu,
    "plain_conv_unet_bilinear_grouped": build_plain_conv_unet_bilinear_grouped,
    "flexible_unet_relu": build_flexible_unet_relu,
    "csrnet_adapter": build_csrnet_adapter,
    "mcnn_adapter": build_mcnn_adapter,
    "resnet_enc": build_resnet_enc,
    "residual_encoder_unet": build_residual_encoder_unet,
}


def build_model(
    model_type: str,
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate a model by *model_type* key.

    Args:
        model_type: One of ``'plain_conv_unet'``, ``'residual_encoder_unet'``.
        input_channels: Number of input image channels.
        num_classes: Number of output density-map channels.
        deep_supervision: If ``True``, produce multi-scale outputs.
        **kwargs: Extra architecture arguments forwarded to the builder.

    Returns:
        Initialised ``nn.Module``.

    Raises:
        ValueError: If *model_type* is not registered.
    """
    builder = _MODEL_REGISTRY.get(model_type)
    if builder is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return builder(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **kwargs,
    )


__all__ = ["build_model", "reset_model_weights"]
