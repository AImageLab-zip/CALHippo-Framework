"""
AdaptiveResNetCounter: a pretrained ResNet encoder truncated at a configurable
depth to produce per-patch cell-count predictions.

The encoder is sliced so that the spatial output resolution equals
``img_size // patch_size_out``.  A 1×1 conv maps encoder features to
``num_classes`` channels.  An optional per-class learnable scalar and a
configurable output activation (default ReLU) ensure non-negative counts.

Supported ``patch_size_out`` values: **4, 8, 16, 32** (determined by the
fixed stride pattern of a standard ResNet).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

# ---- Output-activation look-up ----
_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}


class AdaptiveResNetCounter(nn.Module):
    """Truncated pretrained ResNet for patched density estimation.

    Args:
        num_classes: Number of output density-map channels.
        use_resnet50: If ``True`` use ResNet-50 (Bottleneck blocks);
            otherwise ResNet-18 (BasicBlock).
        patch_size_out: Spatial down-sampling factor relative to the input.
            Must be one of ``{4, 8, 16, 32}``.
        output_scalers: Per-class learnable scalars applied **before** the
            output activation.  Pass a list of *num_classes* initial values
            to enable, or ``None`` to disable (default).
        output_activation: Name of the output activation — ``"relu"``
            (default), ``"softplus"``, or ``"none"``.
    """

    _VALID_PATCHES = {4, 8, 16, 32}

    def __init__(
        self,
        num_classes: int = 3,
        use_resnet50: bool = False,
        patch_size_out: int = 8,
        output_scalers: List[float] | None = None,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()

        if patch_size_out not in self._VALID_PATCHES:
            raise ValueError(
                f"ResNet encoder only supports patch_size_out in "
                f"{sorted(self._VALID_PATCHES)}, got {patch_size_out}."
            )

        # ---- Pretrained backbone ----
        if use_resnet50:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
            channels_map = {4: 256, 8: 512, 16: 1024, 32: 2048}
        else:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            channels_map = {4: 64, 8: 128, 16: 256, 32: 512}

        # ---- Assemble encoder (progressively deeper) ----
        # Base layers always included (stride = 4 total)
        layers: list[nn.Module] = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        ]
        if patch_size_out >= 8:
            layers.append(resnet.layer2)
        if patch_size_out >= 16:
            layers.append(resnet.layer3)
        if patch_size_out >= 32:
            layers.append(resnet.layer4)

        self.encoder = nn.Sequential(*layers)

        # ---- 1×1 mapping to density channels ----
        encoder_out_channels = channels_map[patch_size_out]
        self.out_conv = nn.Conv2d(encoder_out_channels, num_classes, kernel_size=1)

        # ---- Per-class learnable scaler (optional) ----
        if output_scalers is not None:
            init_vals = list(output_scalers)
            if len(init_vals) != num_classes:
                raise ValueError(
                    f"output_scalers length ({len(init_vals)}) != "
                    f"num_classes ({num_classes})"
                )
            self.output_scaler = nn.Parameter(
                torch.tensor(init_vals, dtype=torch.float32)
            )
        else:
            self.output_scaler = None

        # ---- Output activation ----
        act_key = output_activation.lower()
        if act_key not in _OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                f"Choose from: {list(_OUTPUT_ACTIVATIONS.keys())}"
            )
        self.output_act = _OUTPUT_ACTIVATIONS[act_key]()

        logger.debug(
            f"AdaptiveResNetCounter: backbone={'resnet50' if use_resnet50 else 'resnet18'}, "
            f"patch_size_out={patch_size_out}, encoder_channels={encoder_out_channels}, "
            f"num_classes={num_classes}, output_scalers={'enabled' if output_scalers else 'disabled'}, "
            f"output_activation={act_key}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.out_conv(self.encoder(x))
        if self.output_scaler is not None:
            out = out * self.output_scaler.view(1, -1, 1, 1)
        return self.output_act(out)
