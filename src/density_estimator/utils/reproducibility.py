"""Reproducibility helpers for deterministic training runs.

Implements industry-standard seeding for PyTorch, NumPy, and Python's
built-in ``random`` module, following the official PyTorch guidelines:
https://pytorch.org/docs/stable/notes/randomness.html
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility.

    Covers:
    * Python ``random``
    * ``PYTHONHASHSEED`` env-var
    * NumPy global RNG
    * PyTorch CPU & all CUDA devices
    * cuDNN deterministic mode (disables benchmark auto-tuner)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU & current CUDA device
    torch.cuda.manual_seed_all(seed)  # all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed} (deterministic mode enabled)")


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader ``worker_init_fn`` that re-seeds NumPy & random per worker.

    Without this, every worker would share the same NumPy / random state
    after forking, leading to identical augmentation across workers.

    Usage::

        g = torch.Generator()
        g.manual_seed(seed)
        DataLoader(..., worker_init_fn=seed_worker, generator=g)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
