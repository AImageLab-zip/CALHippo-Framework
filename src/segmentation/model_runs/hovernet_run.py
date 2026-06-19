from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from src.segmentation.additional_models.hovernet import HoverNetModel
from src.segmentation.inference.contours_parsing import parse_contours_to_detections
from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection


class HoverNetModelRun(BaseModelRun):
    def __init__(self, model_path: str, params: dict[str, Any]) -> None:
        super().__init__(model_type="hovernet", run_name="HoverNet", params=params)
        self.model_path = model_path

    def load(self) -> None:
        if self.model is not None:
            return

        logger.info(f"Instantiating {self.run_name} from: {self.model_path}")
        start_time = time.perf_counter()
        self.model = HoverNetModel(
            model_path=self.model_path,
            model_mode=self.params.get("model_mode", "original"),
            nr_types=self.params.get("nr_types", None),
            batch_size=self.params.get("batch_size", 32),
            gpu=True,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(f"{self.run_name} instantiation took {elapsed:.2f} seconds")

    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError(f"{self.run_name} has not been loaded.")
        return self.model.eval(crop_img)

    def extract_detections(
        self, mask: np.ndarray, metadata: dict[str, Any]
    ) -> list[Detection]:
        del mask

        # For HoverNet, metadata is expected to contain "inst_info" with contours and "prob" with probabilities for each detected cell.
        # Extract contours and probabilities from them, then parse to Detection objects.

        inst_info = metadata.get("inst_info", {})
        prob_list = metadata.get("prob", [])
        cells_data = []

        for cell_idx, cell_info in inst_info.items():
            cell_contour = cell_info.get("contour")
            if cell_contour is None:
                continue

            prob_index = (
                int(cell_idx) - 1
            )  # Convert to 0-based index for probability lookup
            if prob_index >= len(prob_list):
                continue

            parsed_contour = cell_contour.astype(float).round(0)
            cells_data.append((parsed_contour, float(prob_list[prob_index])))

        detections = parse_contours_to_detections(cells_data, model_name=self.run_name)
        return detections
