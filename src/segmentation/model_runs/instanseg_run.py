from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from instanseg import InstanSeg
from loguru import logger

from src.segmentation.inference.contours_parsing import (
    fast_cell_contours_extraction,
    parse_contours_to_detections,
)
from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection


class InstanSegModelRun(BaseModelRun):
    def __init__(self, model_path: str, params: dict[str, Any]) -> None:
        super().__init__(model_type="instanseg", run_name="InstanSeg", params=params)

        self.model_path = self._resolve_model_path(Path(model_path))

    @staticmethod
    def _resolve_model_path(model_path: Path) -> Path:
        if model_path.is_file():
            return model_path

        checkpoint_path = model_path / "instanseg.pt"
        if model_path.is_dir() and checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError(
            "Invalid InstanSeg model path. Expected a .pt file or a directory "
            f"containing instanseg.pt, got: {model_path}"
        )

    def load(self) -> None:
        if self.model is not None:
            return

        logger.info(f"Instantiating {self.run_name} with model path: {self.model_path}")
        start_time = time.perf_counter()
        
        # Load model
        instanseg_model = torch.jit.load(str(self.model_path))
        self.model = InstanSeg(model_type=instanseg_model, verbosity=1)

        elapsed = time.perf_counter() - start_time
        logger.info(f"{self.run_name} instantiation took {elapsed:.2f} seconds")

    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, None]:
        if self.model is None:
            raise RuntimeError(f"{self.run_name} has not been loaded.")

        # InstantSeg returns the predicted mask and the original image.
        # No metadata is provided.

        mask, _ = self.model.eval_medium_image(crop_img, **self.params)
        mask = mask.cpu().detach().numpy().astype(np.int32)

        # Remove extra dimensions if present (batch and channel)
        if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
            mask = mask[0, 0]

        return mask, None

    def extract_detections(self, mask: np.ndarray, metadata: None) -> list[Detection]:
        del metadata

        # Since no metadata is provided by InstanSeg,
        # directly extract contours from the mask with a default probability.

        cells_data = fast_cell_contours_extraction(mask, default_prob=0.75)
        detections = parse_contours_to_detections(cells_data, model_name=self.run_name)

        return detections
