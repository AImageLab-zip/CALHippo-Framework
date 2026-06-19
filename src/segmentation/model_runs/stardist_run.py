from __future__ import annotations

import time
from typing import Any

import numpy as np
import tensorflow as tf
from csbdeep.utils import normalize
from loguru import logger
from stardist.models import StarDist2D

from src.segmentation.inference.contours_parsing import parse_contours_to_detections
from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection


class StarDistModelRun(BaseModelRun):
    def __init__(self, model_path: str, params: dict[str, Any]) -> None:
        super().__init__(model_type="stardist", run_name="StarDist", params=params)
        self.model_path = model_path

    def load(self) -> None:
        if self.model is not None:
            return

        logger.info(f"Instantiating {self.run_name} from: {self.model_path}")
        start_time = time.perf_counter()
        if self.model_path == "2D_versatile_he":
            self.model = StarDist2D.from_pretrained("2D_versatile_he")
        else:
            self.model = StarDist2D(None, self.model_path)
        elapsed = time.perf_counter() - start_time
        logger.info(f"{self.run_name} instantiation took {elapsed:.2f} seconds")

    def eval(self, crop_img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError(f"{self.run_name} has not been loaded.")

        # Compute n_tiles based on the input image size and a predefined block size
        block_size = self.params.get("block_size", 256)
        n_tiles = (
            max(1, int(np.ceil(crop_img.shape[0] / block_size))),
            max(1, int(np.ceil(crop_img.shape[1] / block_size))),
            1,
        )
        predict_params = {
            key: value for key, value in self.params.items() if key != "block_size"
        }

        mask, metadata = self.model.predict_instances(
            normalize(crop_img), n_tiles=n_tiles, **predict_params
        )

        return mask, metadata

    def extract_detections(
        self, mask: np.ndarray, metadata: dict[str, Any]
    ) -> list[Detection]:

        # For StarDist, metadata is a dict with "prob" and "coord" keys.
        # Extract contours and probabilities from them, then parse to Detection objects.

        mask_shape = mask.shape
        min_x, min_y = 0, 0
        max_x, max_y = mask_shape[1], mask_shape[0]

        del mask

        prob_list = metadata.get("prob", [])
        contours_list = metadata.get("coord", [])
        cells_data = []
        for cell_prob, cell_coords in zip(prob_list, contours_list):
            # Swap axes to get list of couples and reverse to get (x, y) format
            parsed_coords = cell_coords.swapaxes(0, 1)[:, ::-1]
            parsed_coords = parsed_coords.astype(float).round(0)

            # Clip parsed_coords to mask shape
            parsed_coords[:, 0] = np.clip(parsed_coords[:, 0], min_x, max_x - 1)
            parsed_coords[:, 1] = np.clip(parsed_coords[:, 1], min_y, max_y - 1)
            
            cells_data.append((parsed_coords, float(cell_prob)))

        detections = parse_contours_to_detections(cells_data, model_name=self.run_name)
        return detections

    def clean(self) -> None:
        super().clean()
        tf.keras.backend.clear_session()
