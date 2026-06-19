"""
Optimizer factory for ``src.density_estimator``.

Public API::

    from src.density_estimator.optimizers import build_optimizer

    optimizer = build_optimizer("adam", model.parameters(), lr=1e-4)
"""

from __future__ import annotations

from typing import Any, Iterator

import torch.optim as optim
from loguru import logger


def _build_adam(params: Iterator, lr: float, **kwargs: Any) -> optim.Adam:
    """Build ``Adam`` with optional kwargs (betas, weight_decay, eps, …)."""
    accepted = {"betas", "eps", "weight_decay", "amsgrad"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return optim.Adam(params, lr=lr, **filtered)


def _build_adamw(params: Iterator, lr: float, **kwargs: Any) -> optim.AdamW:
    """Build ``AdamW`` with optional kwargs."""
    accepted = {"betas", "eps", "weight_decay", "amsgrad"}
    filtered = {k: float(v) for k, v in kwargs.items() if k in accepted}
    return optim.AdamW(params, lr=lr, **filtered)


def _build_sgd(params: Iterator, lr: float, **kwargs: Any) -> optim.SGD:
    """Build ``SGD`` with optional kwargs (momentum, dampening, …)."""
    accepted = {"momentum", "dampening", "weight_decay", "nesterov"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    for key in ("momentum", "dampening", "weight_decay"):
        if key in filtered:
            filtered[key] = float(filtered[key])
    return optim.SGD(params, lr=lr, **filtered)


# Registry: optimizer_type  →  builder callable
_OPTIMIZER_REGISTRY = {
    "adam": _build_adam,
    "adamw": _build_adamw,
    "sgd": _build_sgd,
}


def build_optimizer(
    optimizer_type: str,
    params: Iterator,
    lr: float,
    **kwargs: Any,
) -> optim.Optimizer:
    """
    Instantiate an optimizer by *optimizer_type* key.

    Args:
        optimizer_type: One of ``'adam'``, ``'adamw'``, ``'sgd'``.
        params: Model parameters (``model.parameters()``).
        lr: Learning rate.
        **kwargs: Extra keyword arguments forwarded to the builder
            (e.g. ``weight_decay``, ``betas``, ``momentum``).

    Returns:
        Configured ``torch.optim.Optimizer``.

    Raises:
        ValueError: If *optimizer_type* is not registered.
    """
    builder = _OPTIMIZER_REGISTRY.get(optimizer_type)
    if builder is None:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Available: {list(_OPTIMIZER_REGISTRY.keys())}"
        )
    opt = builder(params, lr=lr, **kwargs)
    logger.debug(
        f"Optimizer: {optimizer_type} → {opt.__class__.__name__} "
        f"(lr={lr}, kwargs={kwargs})"
    )
    return opt


__all__ = ["build_optimizer"]
