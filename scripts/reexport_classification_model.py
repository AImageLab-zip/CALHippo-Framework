from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
from loguru import logger

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.classification.utils import (
    METADATA_ARTIFACT_NAME,
    METRICS_ARTIFACT_NAME,
    MODEL_ARTIFACT_NAME,
)
from src.utils.logger_setup import setup_logging


def _read_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def reexport_classification_model(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Re-save a classifier bundle with the currently installed sklearn/joblib."""

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    model_path = input_dir / MODEL_ARTIFACT_NAME
    metadata_path = input_dir / METADATA_ARTIFACT_NAME
    metrics_path = input_dir / METRICS_ARTIFACT_NAME

    logger.info(f"Loading classifier model from {model_path}")
    trained_pipeline = joblib.load(model_path)
    metadata = _read_json(metadata_path)

    metadata["runtime_overrides"] = {
        "cli_paths_take_precedence": True,
        "note": (
            "Stored training, test, and output paths are historical metadata from "
            "the original training run. Classification inference uses the CLI "
            "paths passed to src/classification/inference.py."
        ),
    }
    metadata["serialization"] = {
        "scikit_learn_version": version("scikit-learn"),
        "joblib_version": version("joblib"),
        "reexported_from": str(input_dir),
        "reexported_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "This artifact was loaded from the original joblib bundle and re-saved "
            "with the currently installed scikit-learn/joblib versions. Validate "
            "predictions against the original bundle before replacing released "
            "artifacts."
        ),
    }

    logger.info(f"Writing re-exported classifier model to {output_dir}")
    joblib.dump(trained_pipeline, output_dir / MODEL_ARTIFACT_NAME)
    _write_json(output_dir / METADATA_ARTIFACT_NAME, metadata)
    if metrics_path.exists():
        shutil.copy2(metrics_path, output_dir / METRICS_ARTIFACT_NAME)


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export a classification model bundle with current packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/models/classification/ml_classifier"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new_model_classification"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = get_args(argv)
    reexport_classification_model(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
