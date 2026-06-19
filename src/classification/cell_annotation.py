from __future__ import annotations

import numpy as np
from shapely.geometry import shape

from src.classification.data_classes import (
    CellBaseProperties,
    CellDescriptors,
    CellKNNProperties,
    FeatureSpec,
)
from src.classification.utils import (
    angle_diff,
    compute_hu_moments,
    compute_zernike_moments,
    extract_regionprops,
)
from src.utils.helpers import polygon_to_mask


class CellAnnotation:
    """Per-cell container used during feature extraction and inference."""

    def __init__(self, feature_data: dict, image_id: str = "test"):
        self.id = feature_data["id"]
        self.image_id = image_id
        self.polygon = shape(feature_data["geometry"])
        self.centroid = self.polygon.centroid
        self.mask = polygon_to_mask(self.polygon) if self.polygon.is_valid else None
        self.cell_class = (
            feature_data["properties"].get("classification", {}).get("name", None)
        )

        self.base_properties = None
        self.descriptors = None
        self.knn_properties = None
        self.embedding = None
        self.context_embedding = None

        self.predicted_class = None
        self.predicted_probability = None

    def to_feature_dict(
        self,
        feature_spec: FeatureSpec,
        include_class: bool = False,
        include_id: bool = False,
    ) -> dict[str, float | str]:
        """
        Based on the requested feature spec,
        return a dict with the corresponding features for this cell
        """

        feature_dict: dict[str, float | str] = {}

        if include_id:
            feature_dict["id"] = self.id
        if include_class:
            feature_dict["class"] = str(self.cell_class)

        if feature_spec.use_base:
            feature_dict.update(self._base_properties_dict())
        if feature_spec.use_knn:
            feature_dict.update(self._knn_properties_dict())
        if feature_spec.use_descriptors:
            feature_dict.update(self._descriptor_dict())
        if feature_spec.use_embedding:
            feature_dict.update(self._embedding_dict())
        if feature_spec.use_context_embedding:
            feature_dict.update(self._context_embedding_dict())

        return feature_dict

    def _base_properties_dict(self) -> dict[str, float]:
        if self.base_properties is None:
            raise ValueError("Base properties were requested before being computed.")
        return dict(self.base_properties.__dict__)

    def _knn_properties_dict(self) -> dict[str, float]:
        if self.knn_properties is None:
            raise ValueError("KNN properties were requested before being computed.")
        return dict(self.knn_properties.__dict__)

    def _descriptor_dict(self) -> dict[str, float]:
        if self.descriptors is None:
            raise ValueError("Descriptors were requested before being computed.")
        return {
            **{
                f"hu_{i + 1}": value
                for i, value in enumerate(self.descriptors.hu_vectors)
            },
            **{
                f"zernike_{i + 1}": value
                for i, value in enumerate(self.descriptors.zernike_vectors)
            },
        }

    def _embedding_dict(self) -> dict[str, float]:
        if self.embedding is None:
            raise ValueError("Embeddings were requested before being computed.")
        return {f"cell_embed_{i + 1}": value for i, value in enumerate(self.embedding)}

    def _context_embedding_dict(self) -> dict[str, float]:
        if self.context_embedding is None:
            raise ValueError("Context embeddings were requested before being computed.")
        return {
            f"context_embed_{i + 1}": value
            for i, value in enumerate(self.context_embedding)
        }

    def compute_base_properties(self) -> None:
        if self.mask is None:
            self.base_properties = CellBaseProperties(
                area=0.0,
                perimeter=0.0,
                eccentricity=0.0,
                major_axis_length=0.0,
                minor_axis_length=0.0,
                axis_ratio=0.0,
                orientation=0.0,
            )
            return

        props = extract_regionprops(self.mask)
        major_axis_length = props.axis_major_length
        minor_axis_length = props.axis_minor_length
        axis_ratio = (
            minor_axis_length / major_axis_length if major_axis_length != 0 else 0.0
        )
        self.base_properties = CellBaseProperties(
            area=props.area,
            perimeter=props.perimeter,
            eccentricity=props.eccentricity,
            major_axis_length=major_axis_length,
            minor_axis_length=minor_axis_length,
            axis_ratio=axis_ratio,
            orientation=props.orientation,
        )

    def compute_descriptors(self) -> None:
        hu_moments = compute_hu_moments(self.mask)
        zernike_moments = compute_zernike_moments(
            self.polygon,
            image_size=128,
            degree=4,
        )
        self.descriptors = CellDescriptors(
            hu_vectors=hu_moments,
            zernike_vectors=zernike_moments,
        )

    def compute_knn_properties(
        self,
        knn_cells: list[CellAnnotation],
        distances: list[float],
    ) -> None:
        """Compute local spatial features based on the k-nearest neighbor cells."""

        if not knn_cells:
            self.knn_properties = CellKNNProperties(
                mean_distance=0.0,
                std_distance=0.0,
                min_distance=0.0,
                max_distance=0.0,
                orientation_diff=0.0,
                orientation_diff_std=0.0,
                local_density=0.0,
            )
            return

        orientation_diffs = []
        for neighbor_cell in knn_cells:
            orientation_diffs.append(
                angle_diff(
                    self.base_properties.orientation,
                    neighbor_cell.base_properties.orientation,
                )
            )

        distance_array = np.asarray(distances)
        orientation_diff_array = np.asarray(orientation_diffs)
        max_distance = float(np.max(distance_array)) if len(distance_array) > 0 else 0.0

        # Compute local density as number of neighbors
        # divided by area of circle with radius equal to max distance
        local_density = len(knn_cells) / (
            np.pi * (max_distance**2) if max_distance > 0 else 1
        )

        self.knn_properties = CellKNNProperties(
            mean_distance=float(np.mean(distance_array)),
            std_distance=float(np.std(distance_array)),
            min_distance=float(np.min(distance_array)),
            max_distance=max_distance,
            orientation_diff=float(np.mean(orientation_diff_array)),
            orientation_diff_std=float(np.std(orientation_diff_array)),
            local_density=float(local_density),
        )
