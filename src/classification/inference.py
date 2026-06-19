from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from loguru import logger
from shapely.geometry import MultiPolygon

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.classification.backbones import load_backbone
from src.classification.data_classes import FeatureExtractionParams, FeatureSpec
from src.classification.data_loader import (
    get_image_id_from_annotation,
    resolve_test_image_path,
    single_image_data_loader,
)
from src.classification.utils import (
    CLASSES,
    METADATA_ARTIFACT_NAME,
    get_class_ids,
    get_inverse_class_ids,
    load_trained_artifacts,
    str_to_bool,
)
from src.utils.helpers import get_n_available_cpus
from src.utils.logger_setup import setup_logging

_WORKER_CONTEXT: dict[str, object] = {}


def _load_inference_metadata(model_dir: Path) -> dict[str, object]:
    """Load saved classifier metadata without loading the joblib model."""

    metadata_path = model_dir / METADATA_ARTIFACT_NAME
    with metadata_path.open("r") as handle:
        return json.load(handle)


def _uses_uni2h_backbone(metadata: dict[str, object]) -> bool:
    """Return whether saved metadata selects the UNI2-h feature model."""

    config = metadata.get("config", {})
    if not isinstance(config, dict):
        return False
    return str(config.get("feature_model", "")).lower() == "uni2h"


def _load_inference_resources(
    model_dir: Path,
    batch_size: int | None = None,
) -> tuple[object, dict[str, object], FeatureSpec, SimpleNamespace, object | None]:
    """
    Load the saved model and the optional feature backbone.

    Returns:
        trained_pipeline (object): the fitted sklearn pipeline loaded from disk
        metadata (dict[str, object]): the saved metadata
        feature_spec (FeatureSpec): the feature specification
        runtime_config (SimpleNamespace): the runtime configuration
        deep_feature_extraction_model (object | None): the deep feature extraction
            model or None
    """

    # Load the trained pipeline and metadata
    trained_pipeline, metadata, _ = load_trained_artifacts(model_dir)

    # Recreate Namespace (args) from metadata
    config = dict(metadata.get("config", {}))
    runtime_config = SimpleNamespace(**config)
    if batch_size is not None:
        runtime_config.batch_size = batch_size

    # Recover FeatureSpec from metadata
    feature_spec = FeatureSpec.from_dict(metadata["feature_spec"])

    # Load deep model if needed
    deep_feature_extraction_model = load_backbone(
        runtime_config.feature_model,
        feature_spec,
    )
    return (
        trained_pipeline,
        metadata,
        feature_spec,
        runtime_config,
        deep_feature_extraction_model,
    )


def predict_cell_annotations(
    trained_pipeline: object,
    cell_annotations: list,
    feature_spec: FeatureSpec,
    feature_names: list[str],
    astrocyte_area_threshold: float,
) -> list:
    """Run the fitted classifier and attach predictions to each cell."""

    if not cell_annotations:
        return cell_annotations

    astrocyte_class_id = get_class_ids()["Astrocyte"]
    classes_inv = get_inverse_class_ids()

    eval_data = pd.DataFrame(
        [cell.to_feature_dict(feature_spec) for cell in cell_annotations]
    )
    eval_data = eval_data.reindex(columns=feature_names, fill_value=0.0).values

    probabilities = trained_pipeline.predict_proba(eval_data)

    for index, cell in enumerate(cell_annotations):
        if cell.base_properties.area > astrocyte_area_threshold:
            probabilities[index][astrocyte_class_id] = 0.0

        predicted_class_id = int(np.argmax(probabilities[index]))
        cell.predicted_class = classes_inv[predicted_class_id]
        cell.predicted_probability = float(probabilities[index][predicted_class_id])

    return cell_annotations


def export_classified_annotations(
    cell_annotations: list,
    roi_features: list[dict],
    output_folder: Path,
    original_image: np.ndarray,
    image_id: str,
) -> None:
    """Export GeoJSON predictions and a quick overlay visualization."""

    output_folder.mkdir(parents=True, exist_ok=True)

    # Export GeoJSON with predicted classes

    features = []
    for cell in cell_annotations:
        if isinstance(cell.polygon, MultiPolygon):
            logger.warning(
                f"Cell {cell.id} has a MultiPolygon geometry. Skipping export."
            )
            continue

        features.append(
            {
                "type": "Feature",
                "id": cell.id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(cell.polygon.exterior.coords)],
                },
                "properties": {
                    "classification": CLASSES[cell.predicted_class]["geojson_data"],
                    "prediction_probability": cell.predicted_probability,
                },
            }
        )

    output_path = output_folder / f"{image_id}_classification_results.geojson"
    with output_path.open("w") as handle:
        json.dump(
            {"type": "FeatureCollection", "features": roi_features + features},
            handle,
            indent=4,
        )

    # Export thumbnail visualization of predictions

    img_height, img_width = original_image.shape[:2]
    aspect_ratio = img_width / img_height
    base_size = 12
    fig_width, fig_height = (
        (base_size, base_size / aspect_ratio)
        if aspect_ratio >= 1
        else (base_size * aspect_ratio, base_size)
    )

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(original_image, cmap="gray", vmin=0, vmax=255)
    for cell in cell_annotations:
        if isinstance(cell.polygon, MultiPolygon):
            continue
        x_coords, y_coords = cell.polygon.exterior.xy
        plt.fill(
            x_coords,
            y_coords,
            alpha=0.3,
            color=CLASSES[cell.predicted_class]["plt_color"],
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_folder / f"{image_id}_classification_visualization.png", dpi=300)
    plt.close()


def inference_single_image(
    trained_pipeline: object,
    annotation_path: Path,
    image_path: Path,
    output_folder: Path,
    feature_spec: FeatureSpec,
    feature_names: list[str],
    runtime_config: SimpleNamespace,
    deep_feature_extraction_model: object | None,
) -> str:
    """Load one image, classify all cells, and write the outputs."""

    image_id = get_image_id_from_annotation(annotation_path)
    logger.info(f"Processing test image {image_id}...")

    # Wrap extraction params
    feature_extraction_params = FeatureExtractionParams(
        feature_spec=feature_spec,
        knn_k=runtime_config.knn_k,
        batch_size=runtime_config.batch_size,
        crop_padding=runtime_config.crop_padding,
        context_padding=runtime_config.context_padding,
        deep_feature_extraction_model=deep_feature_extraction_model,
    )

    # Extract annotations from the image
    cell_annotations, roi_features, original_image = single_image_data_loader(
        annotation_path,
        image_path,
        feature_extraction_params,
        return_original_image=True,
    )

    # Classify cells
    classified_cell_annotations = predict_cell_annotations(
        trained_pipeline,
        cell_annotations,
        feature_spec,
        feature_names,
        runtime_config.astrocyte_area_threshold,
    )

    # Log class distributions
    class_counts: dict[str, int] = {}
    for cell in classified_cell_annotations:
        class_counts[cell.predicted_class] = (
            class_counts.get(cell.predicted_class, 0) + 1
        )

    total_counts = len(classified_cell_annotations)
    if total_counts > 0:
        logger.info(f"Class distribution for {image_id} in predictions:")
        for class_label, count in class_counts.items():
            percentage = (count / total_counts) * 100
            logger.info(f" - Class {class_label}: {count} samples ({percentage:.2f}%)")

    # Save outputs
    export_classified_annotations(
        classified_cell_annotations,
        roi_features,
        output_folder,
        original_image,
        image_id,
    )

    return image_id


def _init_inference_worker(model_dir: str, batch_size: int | None = None) -> None:
    """Load the saved classifier once per worker process."""

    global _WORKER_CONTEXT

    setup_logging()
    model_path = Path(model_dir)
    (
        trained_pipeline,
        metadata,
        feature_spec,
        runtime_config,
        deep_feature_extraction_model,
    ) = _load_inference_resources(model_path, batch_size=batch_size)

    _WORKER_CONTEXT = {
        "trained_pipeline": trained_pipeline,
        "metadata": metadata,
        "feature_spec": feature_spec,
        "runtime_config": runtime_config,
        "deep_feature_extraction_model": deep_feature_extraction_model,
        "model_dir": model_path,
    }


def _inference_worker(args_tuple: tuple[str, str, str]) -> str:
    """Worker wrapper used by the multiprocessing pool."""

    annotation_path_str, image_path_str, output_folder_str = args_tuple
    return inference_single_image(
        trained_pipeline=_WORKER_CONTEXT["trained_pipeline"],
        annotation_path=Path(annotation_path_str),
        image_path=Path(image_path_str),
        output_folder=Path(output_folder_str),
        feature_spec=_WORKER_CONTEXT["feature_spec"],
        feature_names=_WORKER_CONTEXT["metadata"]["feature_names"],
        runtime_config=_WORKER_CONTEXT["runtime_config"],
        deep_feature_extraction_model=_WORKER_CONTEXT["deep_feature_extraction_model"],
    )


def run_parallel_inference(
    model_dir: Path,
    infer_annotations: list[Path],
    images_folder: Path,
    output_folder: Path,
    batch_size: int | None = None,
):
    num_workers = get_n_available_cpus(exclude_current=True)
    logger.info(f"Using {num_workers} parallel workers for inference.")

    # Since many "inference_single_image" params are shared across all images,
    # the common ones are loaded in the _WORKER_CONTEXT.
    # Here we prepare only the unique params (annotation and image paths) for each
    # single image.

    worker_args = [
        (
            str(annotation_path),
            str(resolve_test_image_path(annotation_path, images_folder)),
            str(output_folder),
        )
        for annotation_path in infer_annotations
    ]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Pool(
        processes=num_workers,
        initializer=_init_inference_worker,
        initargs=(str(model_dir), batch_size),
    ) as pool:
        for index, image_id in enumerate(
            pool.imap_unordered(_inference_worker, worker_args),
            start=1,
        ):
            logger.info(f"Completed {index}/{len(infer_annotations)}: {image_id}")

    return


def run_saved_model_inference(
    model_dir: Path,
    annotations_folder: Path,
    images_folder: Path,
    output_folder: Path,
    parallel_inference: bool = False,
    batch_size: int | None = None,
) -> None:
    """Run the saved model on the given annotation/image folders and save outputs"""

    output_folder.mkdir(parents=True, exist_ok=True)
    infer_annotations = sorted(annotations_folder.glob("*.geojson"))

    if not infer_annotations:
        logger.warning(f"No annotation files found in {annotations_folder}.")
        return

    logger.info(f"Found {len(infer_annotations)} files for inference.")

    metadata = _load_inference_metadata(model_dir)

    # Parallel inference
    if parallel_inference:
        if _uses_uni2h_backbone(metadata):
            logger.warning(
                "Parallel classification inference was requested, but this saved "
                "classifier uses the UNI2-h backbone. UNI2-h is too large to load "
                "once per multiprocessing worker on a single GPU, and parallel mode "
                "would repeatedly initialize the Hugging Face/timm model. Forcing "
                "sequential inference for this run."
            )
        else:
            run_parallel_inference(
                model_dir,
                infer_annotations,
                images_folder,
                output_folder,
                batch_size=batch_size,
            )
            return

    # Sequential inference
    (
        trained_pipeline,
        metadata,
        feature_spec,
        runtime_config,
        deep_feature_extraction_model,
    ) = _load_inference_resources(model_dir, batch_size=batch_size)

    for annotation_path in infer_annotations:
        image_path = resolve_test_image_path(annotation_path, images_folder)

        inference_single_image(
            trained_pipeline=trained_pipeline,
            annotation_path=annotation_path,
            image_path=image_path,
            output_folder=output_folder,
            feature_spec=feature_spec,
            feature_names=metadata["feature_names"],
            runtime_config=runtime_config,
            deep_feature_extraction_model=deep_feature_extraction_model,
        )


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone inference."""

    parser = argparse.ArgumentParser(
        description="Run cell classification inference from a saved model folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_folder", type=Path, required=True)
    parser.add_argument("--annotations_folder", type=Path, required=True)
    parser.add_argument("--images_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, default=None)
    parser.add_argument(
        "--parallel_inference",
        type=str_to_bool,
        default=False,
        help="Enable parallel inference for this run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the saved classifier batch size for inference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    setup_logging()
    output_folder = args.output_folder or args.model_folder

    run_saved_model_inference(
        model_dir=args.model_folder,
        annotations_folder=args.annotations_folder,
        images_folder=args.images_folder,
        output_folder=output_folder,
        parallel_inference=args.parallel_inference,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
