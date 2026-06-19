from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from cellpose.models import CellposeModel
from loguru import logger

from src.segmentation.inference.contours_parsing import (
    fast_cell_contours_extraction,
    parse_contours_to_detections,
)
from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection


def _format_numeric_suffix(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    else:
        logger.warning(
            f"Non-integer diameter value {value} for Cellpose. Rounding to int for run naming."
        )
        return str(int(round(number)))


class CellposeModelRun(BaseModelRun):
    def __init__(self, model_path: str, params: dict[str, Any]) -> None:
        diameter = params.get("diameter", "NA")
        run_name = f"Cellpose_D{_format_numeric_suffix(diameter)}"
        super().__init__(model_type="cellpose", run_name=run_name, params=params)
        self.model_path = model_path

    def load(self) -> None:
        if self.model is not None:
            return

        logger.info(f"Instantiating {self.run_name} from: {self.model_path}")
        start_time = time.perf_counter()
        self.model = CellposeModel(gpu=True, pretrained_model=self.model_path)
        elapsed = time.perf_counter() - start_time
        logger.info(f"{self.run_name} instantiation took {elapsed:.2f} seconds")

    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, list[Any]]:
        if self.model is None:
            raise RuntimeError(f"{self.run_name} has not been loaded.")

        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=torch.cuda.is_available(),
        ):
            mask, metadata, _ = self.model.eval(crop_img, **self.params)

        # Convert any tensor metadata to numpy arrays
        metadata = [
            item.cpu().detach().numpy() if torch.is_tensor(item) else item
            for item in metadata
        ]
        return mask, metadata

    def extract_detections(
        self, mask: np.ndarray, metadata: list[Any]
    ) -> list[Detection]:
        # For Cellpose, metadata[2] contains the logits for the probability mask (shape (H, W) matching the input image)

        logits = metadata[2]
        prob_mask = 1 / (1 + np.exp(-logits))
        cells_data = fast_cell_contours_extraction(mask, prob_mask=prob_mask)
        detections = parse_contours_to_detections(cells_data, model_name=self.run_name)
        return detections
