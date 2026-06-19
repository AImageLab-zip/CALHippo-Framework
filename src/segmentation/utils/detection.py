from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


@dataclass(slots=True)
class Detection:
    """
    Efficiently stores detection metadata and geometry.
    """

    model_name: str
    outline: np.ndarray
    polygon: Polygon
    probability: float = 1.0

    def __post_init__(self):
        # Input Validation
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"Probability {self.probability} is out of bounds (0-1).")

        if not self.polygon.is_valid:
            raise ValueError("The provided polygon geometry is not valid.")

        if self.polygon.is_empty:
            raise ValueError("The provided polygon geometry is empty.")

        if self.polygon.area == 0:
            raise ValueError("The provided polygon has zero area.")

    @property
    def area(self) -> float:
        return self.polygon.area
