"""Legacy-style CSRNet adapter for the density-estimation pipeline.

This keeps the original CSRNet architecture as close as possible to the
upstream `CSRNet-pytorch/model.py` implementation while adapting it to the
current training pipeline.

- supports only `num_classes == 1`
- keeps the original pretrained VGG frontend behavior via `load_weights=False`
- keeps the original linear output by default (`output_activation='none'`)
- upsamples the native /8 CSRNet output back to input resolution while
  preserving the total predicted count
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torchvision import models

from .model_helpers import resize_density_tensor

_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}


class LegacyCSRNetConv2d(nn.Conv2d):
    """Conv2d layer using the legacy CSRNet normal initialisation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pretrained_weight_template: torch.Tensor | None = None
        self._pretrained_bias_template: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def set_pretrained_parameters(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        self._pretrained_weight_template = weight.detach().clone()
        self._pretrained_bias_template = (
            bias.detach().clone() if bias is not None else None
        )
        with torch.no_grad():
            self.weight.copy_(self._pretrained_weight_template)
            if self.bias is not None:
                if self._pretrained_bias_template is not None:
                    self.bias.copy_(self._pretrained_bias_template)
                else:
                    self.bias.zero_()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

        if self._pretrained_weight_template is not None:
            with torch.no_grad():
                self.weight.copy_(self._pretrained_weight_template)
                if self.bias is not None:
                    if self._pretrained_bias_template is not None:
                        self.bias.copy_(self._pretrained_bias_template)
                    else:
                        self.bias.zero_()


def make_layers(
    cfg: list[int | str],
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv2d = LegacyCSRNetConv2d(
            in_channels,
            v,
            kernel_size=3,
            padding=d_rate,
            dilation=d_rate,
        )
        if batch_norm:
            layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv2d, nn.ReLU(inplace=True)])
        in_channels = v
    return nn.Sequential(*layers)


class CSRNetDensityAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        load_weights: bool = False,
        output_activation: str = "none",
        upsample_to_input: bool = True,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if input_channels != 3:
            raise ValueError(
                f"CSRNetDensityAdapter expects input_channels=3, got {input_channels}."
            )
        if num_classes != 1:
            raise ValueError(
                f"CSRNetDensityAdapter only supports num_classes=1, got {num_classes}."
            )

        act_key = output_activation.lower()
        if act_key not in _OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                f"Choose from: {list(_OUTPUT_ACTIVATIONS.keys())}"
            )

        self.seen = 0
        self.load_weights = load_weights
        self.upsample_to_input = upsample_to_input
        self.frontend_feat = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
        ]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(
            self.frontend_feat,
            in_channels=input_channels,
            batch_norm=batch_norm,
        )
        self.backend = make_layers(
            self.backend_feat,
            in_channels=512,
            batch_norm=batch_norm,
            dilation=True,
        )
        self.output_layer = LegacyCSRNetConv2d(64, 1, kernel_size=1)
        self.output_act = _OUTPUT_ACTIVATIONS[act_key]()

        self._initialize_weights()
        if not self.load_weights:
            self._load_pretrained_frontend()

        logger.debug(
            "CSRNetDensityAdapter created: "
            f"load_weights={load_weights}, output_activation={act_key}, "
            f"upsample_to_input={upsample_to_input}, batch_norm={batch_norm}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = self.output_act(x)

        if self.upsample_to_input and x.shape[-2:] != (input_h, input_w):
            x = resize_density_tensor(x, (input_h, input_w))

        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, LegacyCSRNetConv2d):
                module.reset_parameters()
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _load_pretrained_frontend(self) -> None:
        try:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except AttributeError:
            vgg16 = models.vgg16(pretrained=True)

        frontend_convs = [
            module
            for module in self.frontend.modules()
            if isinstance(module, LegacyCSRNetConv2d)
        ]
        pretrained_convs = [
            module
            for module in vgg16.features.modules()
            if isinstance(module, nn.Conv2d)
        ]

        for conv, pretrained in zip(frontend_convs, pretrained_convs, strict=False):
            conv.set_pretrained_parameters(pretrained.weight, pretrained.bias)


def build_csrnet_adapter(
    input_channels: int = 3,
    num_classes: int = 1,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if deep_supervision:
        raise ValueError("CSRNetDensityAdapter does not support deep_supervision=True.")

    load_weights = kwargs.pop("load_weights", False)
    output_activation = kwargs.pop("output_activation", "none")
    upsample_to_input = kwargs.pop("upsample_to_input", True)
    batch_norm = kwargs.pop("batch_norm", False)
    kwargs.pop("use_log_counts", None)

    if kwargs:
        logger.debug(
            f"Ignoring CSRNet adapter kwargs not used by CSRNet: {sorted(kwargs)}"
        )

    return CSRNetDensityAdapter(
        input_channels=input_channels,
        num_classes=num_classes,
        load_weights=load_weights,
        output_activation=output_activation,
        upsample_to_input=upsample_to_input,
        batch_norm=batch_norm,
    )
