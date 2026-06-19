from __future__ import annotations

import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.classification.backbones import load_backbone
from src.classification.data_classes import FeatureExtractionParams
from src.classification.data_loader import load_train_data
from src.classification.inference import run_saved_model_inference
from src.classification.processing_pipelines import (
    build_processing_pipeline,
    get_available_pipeline_names,
)
from src.classification.training import train_pipeline
from src.classification.utils import (
    build_model_metadata,
    load_yaml_config,
    save_trained_artifacts,
    str_to_bool,
)
from src.utils.logger_setup import setup_logging


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally using a flat YAML file for defaults."""

    default_train_annotations_folder = Path("data/input/classification_gt/mixed_new_GT")
    default_train_images_folder = Path("data/input/single_regions/high_res")
    default_test_annotations_folder = Path("data/output/segmentation/RCA4/all_models")
    default_test_images_folder = Path("data/input/single_regions/high_res/RCA4")
    default_output_folder = Path(
        "data/output/classification/RCA4/ml_classifier_logistic_encoder_uni2h"
    )

    default_knn_k = 15
    default_batch_size = 64
    default_crop_padding = 5
    default_context_padding = 50
    default_prediction_pipeline = "logistic_regression"
    default_interneuron_class_weight = 0.7
    default_feature_model = "uni2h"
    default_parallel_inference = False
    default_train_only = False
    default_astrocyte_area_threshold = 100.0

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known_args, _ = config_parser.parse_known_args(argv)
    yaml_config = load_yaml_config(known_args.config)

    parser = ArgumentParser(
        parents=[config_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train_annotations_folder",
        type=Path,
        default=Path(
            yaml_config.get(
                "train_annotations_folder", default_train_annotations_folder
            )
        ),
    )
    parser.add_argument(
        "--train_images_folder",
        type=Path,
        default=Path(
            yaml_config.get(
                "train_images_folder",
                default_train_images_folder,
            )
        ),
    )
    parser.add_argument(
        "--test_annotations_folder",
        type=Path,
        default=Path(
            yaml_config.get(
                "test_annotations_folder", default_test_annotations_folder
            )
        ),
    )
    parser.add_argument(
        "--test_images_folder",
        type=Path,
        default=Path(
            yaml_config.get(
                "test_images_folder",
                default_test_images_folder,
            )
        ),
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path(yaml_config.get("output_folder", default_output_folder)),
        help="Folder used for the saved model artifacts and prediction outputs.",
    )

    parser.add_argument(
        "--knn_k",
        type=int,
        default=yaml_config.get("knn_k", default_knn_k),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=yaml_config.get("batch_size", default_batch_size),
    )
    parser.add_argument(
        "--crop_padding",
        type=int,
        default=yaml_config.get("crop_padding", default_crop_padding),
    )
    parser.add_argument(
        "--context_padding",
        type=int,
        default=yaml_config.get("context_padding", default_context_padding),
    )

    parser.add_argument(
        "--prediction_pipeline",
        type=str,
        default=yaml_config.get("prediction_pipeline", default_prediction_pipeline),
        choices=get_available_pipeline_names(),
    )
    parser.add_argument(
        "--interneuron_class_weight",
        type=float,
        default=yaml_config.get(
            "interneuron_class_weight",
            default_interneuron_class_weight,
        ),
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default=yaml_config.get("feature_model", default_feature_model),
        choices=["resnet18", "uni2h"],
        help=(
            "Deep feature extraction backbone: 'resnet18' or 'uni2h'"
        ),
    )

    parser.add_argument(
        "--parallel_inference",
        type=str_to_bool,
        default=yaml_config.get("parallel_inference", default_parallel_inference),
    )
    parser.add_argument(
        "--train_only",
        type=str_to_bool,
        default=yaml_config.get("train_only", default_train_only),
        help="Skip test-set inference after training when set to true.",
    )
    parser.add_argument(
        "--astrocyte_area_threshold",
        type=float,
        default=yaml_config.get(
            "astrocyte_area_threshold",
            default_astrocyte_area_threshold,
        ),
        help="Set Astrocyte probability to zero for larger cells during inference.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    setup_logging()

    logger.info(
        f"Starting classification training with pipeline '{args.prediction_pipeline}'..."
    )

    # Build the selected pipeline from the registry
    processing_pipeline = build_processing_pipeline(args.prediction_pipeline)

    # Load the deep backbone if needed
    deep_feature_extraction_model = load_backbone(
        args.feature_model,
        processing_pipeline.feature_spec,
    )

    # Group all extraction params into one object
    feature_extraction_params = FeatureExtractionParams(
        feature_spec=processing_pipeline.feature_spec,
        knn_k=args.knn_k,
        batch_size=args.batch_size,
        crop_padding=args.crop_padding,
        context_padding=args.context_padding,
        deep_feature_extraction_model=deep_feature_extraction_model,
    )

    # Load all training data with the requested features
    train_cell_annotations = load_train_data(
        train_annotation_folder=args.train_annotations_folder,
        images_folder=args.train_images_folder,
        feature_extraction_params=feature_extraction_params,
    )

    # Train the model
    trained_pipeline, feature_names, metrics = train_pipeline(
        train_cell_annotations,
        processing_pipeline,
        args.interneuron_class_weight,
    )

    # Save trained model and metadata needed for inference
    metadata = build_model_metadata(
        args,
        trained_pipeline.name,
        trained_pipeline.feature_spec,
    )
    metadata["feature_names"] = feature_names
    save_trained_artifacts(args.output_folder, trained_pipeline, metadata, metrics)

    if args.train_only:
        logger.info(
            "Training finished. Test-set inference skipped because train_only=true."
        )
        return

    # Finally, run inference on the test set using the saved model artifacts
    run_saved_model_inference(
        model_dir=args.output_folder,
        annotations_folder=args.test_annotations_folder,
        images_folder=args.test_images_folder,
        output_folder=args.output_folder,
        parallel_inference=args.parallel_inference,
    )


if __name__ == "__main__":
    main()
