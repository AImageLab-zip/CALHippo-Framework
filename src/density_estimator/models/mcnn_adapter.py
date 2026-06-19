"""Legacy MCNN adapter for the density-estimation pipeline.

This re-implements the original MCNN architecture locally so it can be
registered in the ``src.density_estimator`` model factory without depending
on the legacy script modules.

The adapter is intentionally minimal:

- accepts standard ``build_model(...)`` arguments
- supports only ``num_classes == 1``
- converts multi-channel inputs to a single channel internally
- upsamples the native MCNN /4 output back to input resolution while
  preserving total predicted count
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from .model_helpers import resize_density_tensor

_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}


class LegacyInitConv2d(nn.Conv2d):
    """Conv2d layer using the legacy MCNN normal initialisation."""

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class MCNNConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        same_padding: bool = True,
        relu: bool = True,
        bn: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv = LegacyInitConv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        self.bn = (
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)


class MCNNDensityAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        grayscale_mode: str = "mean",
        output_activation: str = "none",
        bn: bool = False,
    ) -> None:
        super().__init__()

        if num_classes != 1:
            raise ValueError(
                f"MCNNDensityAdapter only supports num_classes=1, got {num_classes}."
            )
        if grayscale_mode not in {"mean", "first"}:
            raise ValueError(
                f"Unsupported grayscale_mode '{grayscale_mode}'. Choose from ['mean', 'first']."
            )

        act_key = output_activation.lower()
        if act_key not in _OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                f"Choose from: {list(_OUTPUT_ACTIVATIONS.keys())}"
            )

        self.input_channels = input_channels
        self.grayscale_mode = grayscale_mode
        self.output_act = _OUTPUT_ACTIVATIONS[act_key]()

        self.branch1 = nn.Sequential(
            MCNNConvBlock(1, 16, 9, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(16, 32, 7, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(32, 16, 7, bn=bn),
            MCNNConvBlock(16, 8, 7, bn=bn),
        )

        self.branch2 = nn.Sequential(
            MCNNConvBlock(1, 20, 7, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(20, 40, 5, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(40, 20, 5, bn=bn),
            MCNNConvBlock(20, 10, 5, bn=bn),
        )

        self.branch3 = nn.Sequential(
            MCNNConvBlock(1, 24, 5, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(24, 48, 3, bn=bn),
            nn.MaxPool2d(2),
            MCNNConvBlock(48, 24, 3, bn=bn),
            MCNNConvBlock(24, 12, 3, bn=bn),
        )

        self.fuse = MCNNConvBlock(30, 1, 1, bn=bn)

        logger.debug(
            "MCNNDensityAdapter created: "
            f"input_channels={input_channels}, grayscale_mode={grayscale_mode}, "
            f"output_activation={act_key}, bn={bn}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        x = self._to_grayscale(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        density = self.fuse(torch.cat((x1, x2, x3), dim=1))
        density = self.output_act(density)

        if density.shape[-2:] != (input_h, input_w):
            density = resize_density_tensor(density, (input_h, input_w))

        return density

    def _to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if self.grayscale_mode == "first":
            return x[:, :1]
        return x.mean(dim=1, keepdim=True)


def build_mcnn_adapter(
    input_channels: int = 3,
    num_classes: int = 1,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if deep_supervision:
        raise ValueError("MCNNDensityAdapter does not support deep_supervision=True.")

    grayscale_mode = kwargs.pop("grayscale_mode", "mean")
    output_activation = kwargs.pop("output_activation", "none")
    bn = kwargs.pop("bn", False)
    kwargs.pop("use_log_counts", None)

    if kwargs:
        logger.debug(f"Ignoring MCNN adapter kwargs not used by MCNN: {sorted(kwargs)}")

    return MCNNDensityAdapter(
        input_channels=input_channels,
        num_classes=num_classes,
        grayscale_mode=grayscale_mode,
        output_activation=output_activation,
        bn=bn,
    )
