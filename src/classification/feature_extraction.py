from __future__ import annotations

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial import KDTree
from torch import nn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.classification.cell_annotation import CellAnnotation
from src.classification.data_classes import DeepFeatureExtractorModel, FeatureSpec


def base_feature_extraction(
    cell_geojson_features: list[dict],
    image_id: str,
    compute_descriptors: bool,
) -> list[CellAnnotation]:
    """Create cell objects and compute the shared geometric features."""

    cell_annotations: list[CellAnnotation] = []
    for feature in cell_geojson_features:
        cell_annotation = CellAnnotation(feature, image_id=image_id)
        if cell_annotation.mask is None:
            logger.error(f"Skipping cell ID {cell_annotation.id} due to invalid mask.")
            continue

        cell_annotation.compute_base_properties()
        if compute_descriptors:
            cell_annotation.compute_descriptors()

        cell_annotations.append(cell_annotation)

    return cell_annotations


def knn_feature_extraction(
    cell_annotations: list[CellAnnotation],
    k: int = 15,
) -> list[CellAnnotation]:
    """Compute local spatial features from the cell centroids."""

    if not cell_annotations:
        return cell_annotations

    if len(cell_annotations) == 1:
        cell_annotations[0].compute_knn_properties([], [])
        return cell_annotations

    centroids = np.array(
        [[cell.centroid.x, cell.centroid.y] for cell in cell_annotations],
        dtype=float,
    )
    kdtree = KDTree(centroids)
    effective_k = min(k, len(cell_annotations) - 1)

    # For each cell, find the neighbours with their distances and compute KNN properties
    for index, cell in enumerate(cell_annotations):
        distances, indices = kdtree.query([centroids[index]], k=effective_k + 1)
        knn_indices = indices[0][1:]
        knn_distances = distances[0][1:]
        knn_cells = [cell_annotations[neighbor_idx] for neighbor_idx in knn_indices]
        cell.compute_knn_properties(knn_cells, knn_distances.tolist())

    return cell_annotations


def extract_embeddings_from_cells(
    cell_annotations: list[CellAnnotation],
    original_image: np.ndarray,
    feature_extractor: nn.Module,
    preprocess: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    padding: int = 5,
) -> list[np.ndarray]:
    """Crop cells from the original image and extract embedding vectors."""

    if not cell_annotations:
        return []

    all_extracted_features: list[np.ndarray] = []

    for batch_start in tqdm(range(0, len(cell_annotations), batch_size)):
        batch_cells = cell_annotations[batch_start : batch_start + batch_size]
        input_tensors = []

        # Crop each cell with padding and apply the preprocessing
        for cell in batch_cells:
            minx, miny, maxx, maxy = cell.polygon.bounds
            minx = max(int(minx) - padding, 0)
            miny = max(int(miny) - padding, 0)
            maxx = min(int(maxx) + padding, original_image.shape[1])
            maxy = min(int(maxy) + padding, original_image.shape[0])

            cell_image = original_image[miny:maxy, minx:maxx]
            pil_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
            pil_image = to_pil_image(pil_image)
            input_tensors.append(preprocess(pil_image))

        # Stack and process the batch
        input_batch = torch.stack(input_tensors).to(device)
        with torch.no_grad():
            features = feature_extractor(input_batch)

        features = features.cpu().view(features.size(0), -1).numpy()

        all_extracted_features.extend(features)

    return all_extracted_features


def deep_feature_extraction(
    cell_annotations: list[CellAnnotation],
    original_image: np.ndarray,
    feature_spec: FeatureSpec,
    deep_feature_extraction_model: DeepFeatureExtractorModel,
    batch_size: int,
    crop_padding: int,
    context_padding: int,
) -> list[CellAnnotation]:
    """
    Extract cell embeddings, and optionally context embeddings, in batches.
    The extraction is done outside the CellAnnotation class in order to parallelize the computation.
    """

    if deep_feature_extraction_model is None:
        raise ValueError("A deep feature extractor is required for embedding features.")

    logger.debug("Extracting cell embeddings...")
    cells_deep_features = extract_embeddings_from_cells(
        cell_annotations,
        original_image,
        deep_feature_extraction_model.feature_extractor,
        deep_feature_extraction_model.preprocess,
        deep_feature_extraction_model.device,
        batch_size=batch_size,
        padding=crop_padding,
    )

    # Assign to each cell the corresponding embedding vector
    for cell, features in zip(cell_annotations, cells_deep_features):
        cell.embedding = features

    if not feature_spec.use_context_embedding:
        return cell_annotations

    logger.debug("Extracting context embeddings...")
    context_deep_features = extract_embeddings_from_cells(
        cell_annotations,
        original_image,
        deep_feature_extraction_model.feature_extractor,
        deep_feature_extraction_model.preprocess,
        deep_feature_extraction_model.device,
        batch_size=batch_size,
        padding=context_padding,
    )

    # Assign to each cell the corresponding context embedding vector
    for cell, context_features in zip(cell_annotations, context_deep_features):
        cell.context_embedding = context_features

    return cell_annotations


def extract_requested_features(
    cell_geojson_features: list[dict],
    image_id: str,
    feature_spec: FeatureSpec,
    knn_k: int,
    original_image: np.ndarray | None,
    deep_feature_extraction_model: DeepFeatureExtractorModel | None,
    batch_size: int,
    crop_padding: int,
    context_padding: int,
) -> list[CellAnnotation]:
    """Extracts from the cell GeoJSON and original image the requested features.
    
    Args:
        cell_geojson_features (list[dict]): List of GeoJSON features for the cells.
        image_id (str): ID of the image (used for annotation metadata).
        feature_spec (FeatureSpec): Specification of which features to extract.
        knn_k (int): Number of neighbors for KNN feature extraction.
        original_image (np.ndarray | None): The original image array, required for deep features.
        deep_feature_extraction_model (DeepFeatureExtractorModel | None): The model to use for deep feature extraction.
        batch_size (int): Batch size for deep feature extraction.
        crop_padding (int): Padding around cell crops for embedding extraction.
        context_padding (int): Padding around cell crops for context embedding extraction.

    Returns:
        list[CellAnnotation]: List of CellAnnotation objects with the requested features extracted.
    """

    cell_annotations = base_feature_extraction(
        cell_geojson_features,
        image_id=image_id,
        compute_descriptors=feature_spec.use_descriptors,
    )

    if feature_spec.use_knn:
        cell_annotations = knn_feature_extraction(cell_annotations, k=knn_k)

    if feature_spec.requires_deep_features:
        if original_image is None:
            raise ValueError("The original image is required for embedding extraction.")
        cell_annotations = deep_feature_extraction(
            cell_annotations,
            original_image,
            feature_spec,
            deep_feature_extraction_model,
            batch_size,
            crop_padding,
            context_padding,
        )

    return cell_annotations
