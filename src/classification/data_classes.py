from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureSpec:
    """Explicit list of which cell features a pipeline needs."""

    use_base: bool = True
    use_descriptors: bool = True
    use_knn: bool = True
    use_embedding: bool = False
    use_context_embedding: bool = False

    @property
    def requires_deep_features(self) -> bool:
        # Automatic check if any embedding features are requested
        return self.use_embedding or self.use_context_embedding

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSpec":
        return cls(**data)



@dataclass
class DeepFeatureExtractorModel:
    """
    Dataclass to hold deep feature extractor model and related components.
    """

    feature_extractor: nn.Module
    preprocess: Callable
    device: torch.device


@dataclass
class FeatureExtractionParams:
    """Dataclass to hold all parameters related to feature extraction."""

    feature_spec: FeatureSpec
    knn_k: int
    deep_feature_extraction_model: DeepFeatureExtractorModel | None
    batch_size: int
    crop_padding: int
    context_padding: int


@dataclass
class CellBaseProperties:
    """Dataclass to hold basic geometric properties of a cell."""

    area: float
    perimeter: float
    eccentricity: float
    major_axis_length: float
    minor_axis_length: float
    axis_ratio: float
    orientation: float


@dataclass
class CellKNNProperties:
    """
    Dataclass to hold k-nearest neighbor properties of a cell.
    """

    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    orientation_diff: float
    orientation_diff_std: float
    local_density: float


@dataclass
class CellDescriptors:
    """Dataclass to hold shape descriptors of a cell."""

    hu_vectors: np.ndarray
    zernike_vectors: np.ndarray
