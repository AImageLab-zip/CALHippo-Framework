from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch.cuda import empty_cache

from src.segmentation.utils.detection import Detection


class BaseModelRun(ABC):
    def __init__(
        self,
        model_type: str,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.model_type = model_type
        self.run_name = run_name
        self.params = params or {}
        self.model: Any | None = None

    @abstractmethod
    def load(self) -> None:
        """Instantiate the underlying model lazily."""

    @abstractmethod
    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, Any]:
        """Run inference on a single crop."""

    @abstractmethod
    def extract_detections(self, mask: np.ndarray, metadata: Any) -> list[Detection]:
        """Convert model outputs into Detection objects."""

    def clean(self) -> None:
        """Release model references and GPU memory."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            empty_cache()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_type={self.model_type!r}, "
            f"run_name={self.run_name!r}, params={self.params!r})"
        )
