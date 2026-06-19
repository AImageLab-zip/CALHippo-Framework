from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from src.classification.cell_annotation import CellAnnotation
from src.classification.data_classes import FeatureExtractionParams
from src.classification.feature_extraction import extract_requested_features
from src.utils.helpers import load_tif_image, split_cell_roi_geojson


def get_image_id_from_annotation(annotation_path: Path) -> str:
    """Map a segmentation GeoJSON filename back to its HR crop id."""

    return annotation_path.stem.replace("_HR_crop_merged", "").replace("_HR_crop", "")


def resolve_train_image_path(annotation_path: Path, images_folder: Path) -> Path:
    """Resolve the training crop path from the labelled GeoJSON filename."""

    filename_parts = annotation_path.stem.split("_")
    if len(filename_parts) < 2:
        raise ValueError(f"Unexpected training annotation name: {annotation_path.name}")

    region = filename_parts[0]
    image_id = filename_parts[1]
    return images_folder / f"{region}-adj" / f"{image_id}_HR_crop.tif"


def resolve_test_image_path(annotation_path: Path, images_folder: Path) -> Path:
    """Resolve the test crop path from the segmentation GeoJSON filename."""

    image_id = get_image_id_from_annotation(annotation_path)
    return images_folder / f"{image_id}_HR_crop.tif"


def single_image_data_loader(
    annotation_path: Path,
    image_path: Path,
    extraction_params: FeatureExtractionParams,
    return_original_image: bool = False,
) -> tuple[list[CellAnnotation], list[dict], np.ndarray | None]:
    """Load one GeoJSON/image pair and extract the requested cell features."""

    logger.info(
        f"Loading data from {annotation_path} for image ID {image_path.stem}..."
    )

    # Load GeoJSON data and slit cell and ROI features
    with annotation_path.open("r") as handle:
        geojson_data = json.load(handle)
    cell_features, roi_features = split_cell_roi_geojson(geojson_data)

    # Load the original image if needed
    original_image = None
    if return_original_image or extraction_params.feature_spec.requires_deep_features:
        original_image = load_tif_image(image_path)

    # Extract the requested features for each cell
    cell_annotations = extract_requested_features(
        cell_geojson_features=cell_features,
        image_id=image_path.stem,
        feature_spec=extraction_params.feature_spec,
        knn_k=extraction_params.knn_k,
        original_image=original_image,
        deep_feature_extraction_model=extraction_params.deep_feature_extraction_model,
        batch_size=extraction_params.batch_size,
        crop_padding=extraction_params.crop_padding,
        context_padding=extraction_params.context_padding,
    )

    return cell_annotations, roi_features, original_image


def load_train_data(
    train_annotation_folder: Path,
    images_folder: Path,
    feature_extraction_params: FeatureExtractionParams,
) -> list[CellAnnotation]:
    """Load and aggregate all labelled training annotations."""

    full_cell_annotations: list[CellAnnotation] = []

    for annotation_file in sorted(train_annotation_folder.glob("*.geojson")):
        
        image_path = resolve_train_image_path(annotation_file, images_folder)

        # Extract features for the cells in this image
        cell_annotations, _, _ = single_image_data_loader(
            annotation_file,
            image_path,
            feature_extraction_params,
            return_original_image=False,
        )

        # Append the annotations to the full list
        full_cell_annotations.extend(cell_annotations)

    return full_cell_annotations
