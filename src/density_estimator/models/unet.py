"""
UNet model builders for density estimation.

Supports ``PlainConvUNet`` and ``ResidualEncoderUNet`` from the
``dynamic_network_architectures`` library.  Architecture hyper-parameters
are fully configurable via the ``**kwargs`` passthrough so that users can
override them in YAML without touching Python code.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from loguru import logger

from .flexible_unet import FlexibleUNet2D, FlexibleUNet2DReLU
from .resnet_encoder import AdaptiveResNetCounter
from .unet_bilinear_grouped import PlainConvUNetBilinearGrouped

# ---- Output-activation look-up (shared by wrappers in this file) ----
_OUTPUT_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=False),
    "softplus": lambda: nn.Softplus(),
    "none": lambda: nn.Identity(),
}

# ============================================================
# String-to-class look-ups for entries that arrive as strings
# from YAML / argparse.
# ============================================================

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

_CONV_OPS: Dict[str, type] = {
    "Conv2d": nn.Conv2d,
}


def _resolve(mapping: Dict[str, type], key: Any, label: str) -> type:
    """
    Resolve *key* to a ``nn.Module`` class.

    If *key* is already a type it is returned as-is; if it is a string
    present in *mapping* the corresponding class is returned.
    """
    if isinstance(key, type):
        return key
    if isinstance(key, str) and key in mapping:
        return mapping[key]
    raise ValueError(f"Unknown {label} '{key}'. Choose from: {list(mapping.keys())}")


# ============================================================
# Builders
# ============================================================


def build_plain_conv_unet(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> PlainConvUNet:
    """
    Build a ``PlainConvUNet`` with sensible defaults, overridable via
    ``kwargs`` (e.g. from the YAML ``MODEL.kwargs`` section).

    Returns:
        Initialised model instance.
    """
    params: Dict[str, Any] = {
        "n_stages": 4,
        "features_per_stage": (16, 32, 64, 128),
        "strides": (1, 2, 2, 2),
        "n_conv_per_stage": (2, 2, 2, 2),
        "n_conv_per_stage_decoder": (2, 2, 2),
        "kernel_sizes": (3, 3, 3, 3),
        "conv_bias": True,
        "conv_op": nn.Conv2d,
        "norm_op": nn.InstanceNorm2d,
        "dropout_op": None,
        "nonlin": nn.LeakyReLU,
    }
    params.update(kwargs)

    # Resolve string references coming from YAML
    params["conv_op"] = _resolve(_CONV_OPS, params["conv_op"], "conv_op")
    params["norm_op"] = _resolve(_NORM_OPS, params["norm_op"], "norm_op")
    if params.get("nonlin") is not None:
        params["nonlin"] = _resolve(_NONLINS, params["nonlin"], "nonlin")

    model = PlainConvUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **params,
    )
    model.initialize(model)

    logger.debug(
        f"PlainConvUNet created: in={input_channels}, out={num_classes}, "
        f"stages={params['n_stages']}, deep_supervision={deep_supervision}"
    )
    return model


def build_residual_encoder_unet(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> ResidualEncoderUNet:
    """
    Build a ``ResidualEncoderUNet`` with sensible defaults, overridable via
    ``kwargs``.

    Returns:
        Initialised model instance.
    """
    params: Dict[str, Any] = {
        "n_stages": 4,
        "features_per_stage": (16, 32, 64, 128),
        "strides": (1, 2, 2, 2),
        "n_blocks_per_stage": (2, 2, 2, 2),
        "n_conv_per_stage_decoder": (2, 2, 2),
        "kernel_sizes": (3, 3, 3, 3),
        "conv_bias": True,
        "conv_op": nn.Conv2d,
        "norm_op": nn.InstanceNorm2d,
        "dropout_op": None,
        "nonlin": nn.LeakyReLU,
    }
    params.update(kwargs)

    params["conv_op"] = _resolve(_CONV_OPS, params["conv_op"], "conv_op")
    params["norm_op"] = _resolve(_NORM_OPS, params["norm_op"], "norm_op")
    if params.get("nonlin") is not None:
        params["nonlin"] = _resolve(_NONLINS, params["nonlin"], "nonlin")

    model = ResidualEncoderUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **params,
    )
    model.initialize(model)

    logger.debug(
        f"ResidualEncoderUNet created: in={input_channels}, out={num_classes}, "
        f"stages={params['n_stages']}, deep_supervision={deep_supervision}"
    )
    return model


class PlainConvUNetReLU(nn.Module):
    """``PlainConvUNet`` with optional per-class learnable scaler and
    configurable output activation (default ReLU).

    Density maps are non-negative by definition, so the activation ensures
    no spurious negative predictions.
    """

    def __init__(
        self,
        base: PlainConvUNet,
        num_classes: int = 3,
        output_scalers: list[float] | None = None,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.base = base

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

    def forward(self, x):
        out = self.base(x)
        if self.output_scaler is not None:
            out = out * self.output_scaler.view(1, -1, 1, 1)
        return self.output_act(out)

    # Expose .encoder / .decoder so that reset_model_weights still works
    # when called via model.apply(...).


def build_plain_conv_unet_relu(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> PlainConvUNetReLU:
    """Build a ``PlainConvUNet`` with configurable output activation + scaler.

    Extra kwargs consumed here (not forwarded to PlainConvUNet):
        ``output_scalers``, ``output_activation``.
    """
    # Pop wrapper-only keys before forwarding to PlainConvUNet
    output_scalers = kwargs.pop("output_scalers", None)
    output_activation = kwargs.pop("output_activation", "relu")
    kwargs.pop("use_log_counts", None)  # pipeline-only

    base = build_plain_conv_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **kwargs,
    )
    model = PlainConvUNetReLU(
        base,
        num_classes=num_classes,
        output_scalers=output_scalers,
        output_activation=output_activation,
    )
    logger.debug(
        f"PlainConvUNetReLU created: in={input_channels}, out={num_classes}, "
        f"deep_supervision={deep_supervision}, "
        f"output_scalers={'enabled' if output_scalers else 'disabled'}, "
        f"output_activation={output_activation}"
    )
    return model


def build_flexible_unet_relu(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> FlexibleUNet2DReLU:
    """Build a ``FlexibleUNet2D`` with configurable output activation + scaler.

    Relevant *kwargs* (others are silently ignored):
        ``n_stages``, ``img_size``, ``patch_size_out``, ``size``,
        ``norm_op``, ``nonlin``, ``output_scalers``, ``output_activation``.
    """
    # ---- Wrapper-only keys ----
    output_scalers = kwargs.pop("output_scalers", None)
    output_activation = kwargs.pop("output_activation", "relu")

    # Extract only the keys FlexibleUNet2D understands; ignore the rest
    # (default model_kwargs from config.py may include PlainConvUNet-only keys).
    accepted = {"n_stages", "img_size", "patch_size_out", "size", "norm_op", "nonlin"}
    # Keys consumed by the trainer pipeline, not by the model itself:
    _pipeline_only = {"use_log_counts"}
    flex_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _pipeline_only:
            continue  # skip pipeline-only keys
        if k in accepted:
            key = "depth" if k == "n_stages" else k  # rename n_stages→depth
            flex_kw[key] = v

    # Resolve string ↔ class for norm / nonlin (same tables as PlainConvUNet)
    if "norm_op" in flex_kw and isinstance(flex_kw["norm_op"], str):
        flex_kw["norm_op"] = _resolve(_NORM_OPS, flex_kw["norm_op"], "norm_op")
    if "nonlin" in flex_kw and isinstance(flex_kw["nonlin"], str):
        flex_kw["nonlin"] = _resolve(_NONLINS, flex_kw["nonlin"], "nonlin")

    base = FlexibleUNet2D(
        in_channels=input_channels,
        num_classes=num_classes,
        **flex_kw,
    )
    model = FlexibleUNet2DReLU(
        base,
        output_scalers=output_scalers,
        output_activation=output_activation,
    )
    logger.debug(
        f"FlexibleUNet2DReLU created: in={input_channels}, out={num_classes}, "
        f"kwargs={flex_kw}, output_scalers={'enabled' if output_scalers else 'disabled'}, "
        f"output_activation={output_activation}"
    )
    return model


def build_resnet_enc(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> AdaptiveResNetCounter:
    """Build an ``AdaptiveResNetCounter`` for patched density estimation.

    Relevant *kwargs* (others are silently ignored):
        ``patch_size_out``, ``use_resnet50``, ``output_scalers``,
        ``output_activation``.
    """
    accepted = {"patch_size_out", "use_resnet50", "output_scalers", "output_activation"}
    _pipeline_only = {"use_log_counts"}
    enc_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _pipeline_only:
            continue
        if k in accepted:
            enc_kw[k] = v

    model = AdaptiveResNetCounter(
        num_classes=num_classes,
        **enc_kw,
    )
    logger.debug(
        f"AdaptiveResNetCounter created: num_classes={num_classes}, kwargs={enc_kw}"
    )
    return model


def build_plain_conv_unet_bilinear_grouped(
    input_channels: int = 3,
    num_classes: int = 3,
    deep_supervision: bool = False,
    **kwargs: Any,
) -> PlainConvUNetBilinearGrouped:
    """Build a ``PlainConvUNetBilinearGrouped``.

    Uses bilinear upsampling in the decoder and a grouped 1×1 final
    convolution (``groups=num_classes``) so each output channel has
    a strictly independent pathway.  Output is ReLU-activated.
    """
    # Pop wrapper-only keys before forwarding
    output_scalers = kwargs.pop("output_scalers", None)
    output_activation = kwargs.pop("output_activation", "relu")
    kwargs.pop("use_log_counts", None)  # pipeline-only

    params: Dict[str, Any] = {
        "n_stages": 4,
        "features_per_stage": (16, 32, 64, 128),
        "strides": (1, 2, 2, 2),
        "n_conv_per_stage": (2, 2, 2, 2),
        "n_conv_per_stage_decoder": (2, 2, 2),
        "kernel_sizes": (3, 3, 3, 3),
        "conv_bias": True,
        "conv_op": nn.Conv2d,
        "norm_op": nn.InstanceNorm2d,
        "dropout_op": None,
        "nonlin": nn.LeakyReLU,
    }
    params.update(kwargs)

    # Resolve string references coming from YAML
    params["conv_op"] = _resolve(_CONV_OPS, params["conv_op"], "conv_op")
    params["norm_op"] = _resolve(_NORM_OPS, params["norm_op"], "norm_op")
    if params.get("nonlin") is not None:
        params["nonlin"] = _resolve(_NONLINS, params["nonlin"], "nonlin")

    model = PlainConvUNetBilinearGrouped(
        input_channels=input_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        output_scalers=output_scalers,
        output_activation=output_activation,
        **params,
    )
    model.initialize(model)

    logger.debug(
        f"PlainConvUNetBilinearGrouped created: in={input_channels}, "
        f"out={num_classes}, stages={params['n_stages']}, "
        f"deep_supervision={deep_supervision}, "
        f"output_scalers={'enabled' if output_scalers else 'disabled'}, "
        f"output_activation={output_activation}"
    )
    return model
