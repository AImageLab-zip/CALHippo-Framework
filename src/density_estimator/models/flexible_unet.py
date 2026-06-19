"""
FlexibleUNet2D: a lightweight UNet whose decoder can stop early to produce
output at a *sub-resolution* grid (e.g. 32×32 for 128×128 input).

This is the architecture used by the **patched density estimation** pipeline.
The model is designed for targets that have been:

1. **Patchified** — ``avg_pool2d(mask, patch_size) * patch_size²`` gathers per-
   patch cell counts.
2. **Log-transformed** — ``log1p(patched)`` compresses dynamic range.

The output has shape ``(B, C, img_size // patch_size_out, img_size // patch_size_out)``
in log-count space.  Applying ``torch.expm1`` recovers the actual per-patch counts.

Configurable normalisation (``BatchNorm2d``, ``InstanceNorm2d``, …) and
activation (``ReLU``, ``LeakyReLU``, …) via string or ``nn.Module`` type.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Type, Union

import torch
import torch.nn as nn
from loguru import logger

# ---- look-up tables (reuse the unet.py convention) ----------------------

_NORM_OPS: Dict[str, type] = {
    "InstanceNorm2d": nn.InstanceNorm2d,
    "BatchNorm2d": nn.BatchNorm2d,
}

_NONLINS: Dict[str, type] = {
    "LeakyReLU": nn.LeakyReLU,
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "ELU": nn.ELU,
    "SiLU": nn.SiLU,
}


def _resolve(mapping: Dict[str, type], key: Any, label: str) -> type:
    if isinstance(key, type):
        return key
    if isinstance(key, str) and key in mapping:
        return mapping[key]
    raise ValueError(f"Unknown {label} '{key}'. Choose from: {list(mapping.keys())}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv2d → Norm → Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        conv_kwargs: dict | None = None,
    ):
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1}
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
            norm_op(out_channels),
            nonlin(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownConvBlock(nn.Module):
    """Two (or more) ConvBlocks followed by 2×2 MaxPool."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(ic, oc, norm_op, nonlin)
                for ic, oc in zip(in_channels, out_channels)
            ]
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        for blk in self.conv_blocks:
            x = blk(x)
        return self.pool(x), x  # (pooled, skip)


class UpConvBlock(nn.Module):
    """Two (or more) ConvBlocks with optional ConvTranspose2d upsample."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        up_conv: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm2d,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(ic, oc, norm_op, nonlin)
                for ic, oc in zip(in_channels, out_channels)
            ]
        )
        self.up_conv = up_conv
        if up_conv:
            self.up_conv_op = nn.ConvTranspose2d(
                out_channels[-1],
                out_channels[-1],
                kernel_size=2,
                stride=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.conv_blocks:
            x = blk(x)
        if self.up_conv:
            x = self.up_conv_op(x)
        return x


# ---------------------------------------------------------------------------
# FlexibleUNet2D
# ---------------------------------------------------------------------------


class FlexibleUNet2D(nn.Module):
    """UNet with a *truncated decoder* that outputs at ``img_size // patch_size_out``.

    The encoder always downsamples ``depth`` times (each by 2×).  The decoder
    only upsamples enough times to reach the target output resolution.  Skip
    connections from the encoder are consumed whenever the decoder passes
    through the matching spatial level.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    num_classes : int
        Number of output channels (cell types).
    size : int
        Base feature width.  Encoder doubles features at each stage.
    depth : int
        Number of encoder down-sampling stages.
    img_size : int
        Spatial size of the input image.
    patch_size_out : int
        Spatial patch size of each output pixel (output is ``img_size // patch_size_out``).
    norm_op : type or str
        Normalisation layer class (``nn.BatchNorm2d``, ``nn.InstanceNorm2d``, or
        their string names).
    nonlin : type or str
        Activation layer class (``nn.LeakyReLU``, ``nn.ReLU``, … or string).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        size: int = 16,
        depth: int = 3,
        img_size: int = 128,
        patch_size_out: int = 4,
        norm_op: Union[Type[nn.Module], str] = nn.InstanceNorm2d,
        nonlin: Union[Type[nn.Module], str] = nn.LeakyReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.size = size
        self.depth = depth
        self.patch_size_out = patch_size_out

        # Resolve string ↔ class
        if isinstance(norm_op, str):
            norm_op = _resolve(_NORM_OPS, norm_op, "norm_op")
        if isinstance(nonlin, str):
            nonlin = _resolve(_NONLINS, nonlin, "nonlin")

        # ------- spatial math -------
        self.target_res = img_size // patch_size_out  # e.g. 32
        self.bott_res = img_size // (2**depth)  # e.g. 16
        self.total_upsamples = int(math.log2(self.target_res / self.bott_res))

        # ------- encoder -------
        self.encoder = nn.ModuleDict()
        self.encoder["0"] = DownConvBlock(
            [in_channels, size],
            [size, size * 2],
            norm_op,
            nonlin,
        )
        for i in range(1, depth):
            self.encoder[str(i)] = DownConvBlock(
                [size * (2**i), size * (2**i)],
                [size * (2**i), size * (2 ** (i + 1))],
                norm_op,
                nonlin,
            )

        # ------- bottleneck + dynamic decoder -------
        self.decoder_blocks = nn.ModuleList()
        self.final_mix: nn.Module | None = None

        if self.total_upsamples > 0:
            self.bottleneck = UpConvBlock(
                [size * (2**depth), size * (2**depth)],
                [size * (2**depth), size * (2 ** (depth + 1))],
                up_conv=True,
                norm_op=norm_op,
                nonlin=nonlin,
            )
            prev_out_ch = size * (2 ** (depth + 1))
            current_level = depth - 1

            for _ in range(self.total_upsamples - 1):
                skip_ch = size * (2 ** (current_level + 1))
                in_ch = prev_out_ch + skip_ch
                out_ch = skip_ch
                self.decoder_blocks.append(
                    UpConvBlock(
                        [in_ch, out_ch],
                        [out_ch, out_ch],
                        up_conv=True,
                        norm_op=norm_op,
                        nonlin=nonlin,
                    )
                )
                prev_out_ch = out_ch
                current_level -= 1

            # final mix (no upsampling)
            skip_ch = size * (2 ** (current_level + 1))
            in_ch = prev_out_ch + skip_ch
            out_ch = skip_ch
            self.final_mix = UpConvBlock(
                [in_ch, out_ch],
                [out_ch, out_ch],
                up_conv=False,
                norm_op=norm_op,
                nonlin=nonlin,
            )
            final_out_channels = out_ch
        else:
            logger.warning(
                "FlexibleUNet2D: target resolution equals bottleneck resolution — "
                "no decoder upsampling needed.  Skip connections are not used."
            )
            self.bottleneck = UpConvBlock(
                [size * (2**depth), size * (2**depth)],
                [size * (2**depth), size * (2 ** (depth + 1))],
                up_conv=False,
                norm_op=norm_op,
                nonlin=nonlin,
            )
            final_out_channels = size * (2 ** (depth + 1))

        self.out_layer = nn.Conv2d(final_out_channels, num_classes, kernel_size=1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_list: list[torch.Tensor] = []

        # encoder
        out, feat = self.encoder["0"](x)
        feat_list.append(feat)
        for key in list(self.encoder)[1:]:
            out, feat = self.encoder[key](out)
            feat_list.append(feat)

        # bottleneck
        out = self.bottleneck(out)

        # decoder with skip connections
        if self.total_upsamples > 0:
            current_level = self.depth - 1
            for blk in self.decoder_blocks:
                out = torch.cat((out, feat_list[current_level]), dim=1)
                out = blk(out)
                current_level -= 1
            out = torch.cat((out, feat_list[current_level]), dim=1)
            out = self.final_mix(out)

        out = self.out_layer(out)
        return out


# ---------------------------------------------------------------------------
# Wrapper that adds output scaler + configurable activation
# ---------------------------------------------------------------------------

# ---- Output-activation look-up ----
_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}


class FlexibleUNet2DReLU(nn.Module):
    """``FlexibleUNet2D`` with optional per-class learnable scaler and
    configurable output activation (default ReLU).

    Args:
        base: The underlying ``FlexibleUNet2D``.
        output_scalers: Per-class learnable scalars applied **before** the
            activation.  Pass a list of *num_classes* floats to enable,
            or ``None`` to disable (default).
        output_activation: ``"relu"`` (default), ``"softplus"``, or
            ``"none"``.
    """

    def __init__(
        self,
        base: FlexibleUNet2D,
        output_scalers: list[float] | None = None,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.base = base

        # ---- Per-class learnable scaler (optional) ----
        num_classes = base.out_channels
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.output_scaler is not None:
            out = out * self.output_scaler.view(1, -1, 1, 1)
        return self.output_act(out)
