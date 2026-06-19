from __future__ import annotations

from typing import Any

import numpy as np

from src.segmentation.additional_models.adaptive_threshold import AdaptiveThresholdModel
from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection


class AdaptiveThresholdModelRun(BaseModelRun):
    def __init__(self, params: dict[str, Any]) -> None:
        run_name = (
            f"ATM_{params.get('method', 'cv2')}_"
            f"{params.get('window_size', 'NA')}_"
            f"{params.get('second_param', 'NA')}"
        )
        super().__init__(model_type="adaptive", run_name=run_name, params=params)

    def load(self) -> None:
        if self.model is None:
            self.model = AdaptiveThresholdModel(**self.params)

    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        if self.model is None:
            raise RuntimeError(f"{self.run_name} has not been loaded.")

        mask, metadata = self.model.eval(crop_img)

        return mask, metadata

    def extract_detections(
        self, mask: np.ndarray, metadata: list[Detection]
    ) -> list[Detection]:
        # ATM directly return the list of Detection objects as metadata

        del mask
        return metadata
