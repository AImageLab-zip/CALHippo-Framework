"""
PlainConvUNet variant with **bilinear upsampling** in the decoder and a
**grouped 1×1 final convolution** so that each output density-map channel
is produced by a strictly independent pathway.

Output is passed through a configurable activation (default ReLU) with an
optional per-class learnable scaler.  Density maps are non-negative.
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

# ---- Output-activation look-up ----
_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}


class BilinearGroupedDecoder(nn.Module):
    """Decoder that uses **bilinear upsampling + 1×1 conv** instead of
    transposed convolutions, and a **grouped** final segmentation layer.

    Parameters
    ----------
    encoder : PlainConvEncoder
        The encoder whose skip connections feed this decoder.
    num_classes : int
        Number of output channels (density-map classes).
    n_conv_per_stage : list[int]
        Number of conv blocks per decoder stage (length = n_stages - 1).
    deep_supervision : bool
        Produce multi-scale outputs when ``True``.
    channels_before_final : int | None
        Intermediate channel count right before the grouped 1×1 conv.
        Must be divisible by *num_classes*.  When ``None`` a sensible
        default is chosen automatically.
    nonlin_first : bool
        ``conv → nonlin → norm`` ordering when ``True``.
    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
    nonlin_kwargs, conv_bias
        Override the encoder defaults when not ``None``.
    """

    def __init__(
        self,
        encoder: PlainConvEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        channels_before_final: int | None = None,
        nonlin_first: bool = False,
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Type[_DropoutNd] | None = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] | None = None,
        nonlin_kwargs: dict | None = None,
        conv_bias: bool | None = None,
        mode: str = "nearest",
    ):
        super().__init__()
        self.mode = mode
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        # Inherit encoder settings where caller didn't override.
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = (
            encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        )
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = (
            encoder.dropout_op_kwargs
            if dropout_op_kwargs is None
            else dropout_op_kwargs
        )
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = (
            encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        )

        # --- Determine channels_before_final ---
        # Must be divisible by num_classes for the grouped conv.
        top_features = encoder.output_channels[0]  # e.g. 16
        if channels_before_final is None:
            # Round up to the nearest multiple of num_classes.
            channels_before_final = int(
                math.ceil(top_features / num_classes) * num_classes
            )
        assert channels_before_final % num_classes == 0, (
            f"channels_before_final ({channels_before_final}) must be "
            f"divisible by num_classes ({num_classes})"
        )
        self.channels_before_final = channels_before_final

        # --- Build stages ---
        stages: list[nn.Module] = []
        upsample_convs: list[nn.Module] = []  # 1×1 conv after bilinear upsample
        seg_layers: list[nn.Module] = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]

            # Bilinear upsample keeps channels unchanged, so we add a 1×1
            # conv to project from input_features_below → input_features_skip.
            upsample_convs.append(
                nn.Conv2d(
                    input_features_below,
                    input_features_skip,
                    kernel_size=1,
                    bias=conv_bias,
                )
            )

            # After concat with skip: 2 × input_features_skip channels.
            is_last_stage = s == (n_stages_encoder - 1)
            stage_out_channels = (
                channels_before_final if is_last_stage else input_features_skip
            )

            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    stage_out_channels,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )

            # Seg layer: grouped 1×1 conv at the last stage, regular
            # elsewhere (for deep supervision compatibility).
            if is_last_stage:
                seg_layers.append(
                    nn.Conv2d(
                        channels_before_final,
                        num_classes,
                        kernel_size=1,
                        groups=num_classes,
                        bias=True,
                    )
                )
            else:
                seg_layers.append(
                    encoder.conv_op(
                        input_features_skip, num_classes, 1, 1, 0, bias=True
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.upsample_convs = nn.ModuleList(upsample_convs)
        self.seg_layers = nn.ModuleList(seg_layers)

        # Upsampling scale factors (derived from encoder strides).
        self._scale_factors: list[tuple[int, ...]] = []
        for s in range(1, n_stages_encoder):
            stride = encoder.strides[-s]
            if isinstance(stride, int):
                stride = [stride, stride]
            self._scale_factors.append(tuple(stride))

    # ------------------------------------------------------------------

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        lres_input = skips[-1]
        seg_outputs: list[torch.Tensor] = []

        for s in range(len(self.stages)):
            # 1. Upsample + 1×1 channel projection.
            scale = self._scale_factors[s]
            interp_kwargs: dict[str, Any] = dict(scale_factor=scale, mode=self.mode)
            if self.mode not in ("nearest", "nearest-exact"):
                interp_kwargs["align_corners"] = False
            x = F.interpolate(lres_input, **interp_kwargs)
            x = self.upsample_convs[s](x)

            # 2. Concat with encoder skip.
            x = torch.cat((x, skips[-(s + 2)]), dim=1)

            # 3. Conv blocks.
            x = self.stages[s](x)

            # 4. Segmentation head.
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# ======================================================================
# Full model wrapper
# ======================================================================


class PlainConvUNetBilinearGrouped(nn.Module):
    """``PlainConvEncoder`` + ``BilinearGroupedDecoder`` + output activation.

    Drop-in replacement for ``PlainConvUNet`` / ``PlainConvUNetReLU`` that
    uses bilinear upsampling instead of transposed convolutions and a
    grouped 1×1 final convolution for class-independent output pathways.

    Supports an optional per-class learnable scaler and configurable output
    activation (default ReLU).
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: Type[_DropoutNd] | None = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] | None = None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        channels_before_final: int | None = None,
        upsampling_mode: str = "nearest",
        output_scalers: list[float] | None = None,
        output_activation: str = "relu",
    ):
        super().__init__()

        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )

        self.decoder = BilinearGroupedDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            channels_before_final=channels_before_final,
            nonlin_first=nonlin_first,
            mode=upsampling_mode,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        out = self.decoder(skips)
        if self.output_scaler is not None:
            out = out * self.output_scaler.view(1, -1, 1, 1)
        return self.output_act(out)

    # Convenience for weight initialisation (same pattern as PlainConvUNet).
    @staticmethod
    def initialize(module: nn.Module) -> None:
        InitWeights_He(1e-2)(module)
