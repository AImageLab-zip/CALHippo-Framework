from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_IMAGE_IDS = ["3305", "3348"]
DEFAULT_REGION = "RCA3"
DEFAULT_CLASSIFICATION_EXPERIMENT = "ml_classifier_logistic_encoder_uni2h"
DEFAULT_SEGMENTATION_EXPERIMENT = "all_models_smoke"
DEFAULT_DATASET_NAME = "allCA_128_96_smooth_b05_k5_roi"
CA_REGIONS = ["RCA1", "RCA2", "RCA3", "RCA4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the maintained two-WSI smoke test through density creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_temp"),
        help="Isolated data root used for the smoke test.",
    )
    parser.add_argument(
        "--image-ids",
        nargs="+",
        default=DEFAULT_IMAGE_IDS,
        help="At least two distinct image IDs for GroupShuffleSplit validation.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        choices=CA_REGIONS,
        help="Single region to segment, classify, and use for density creation.",
    )
    parser.add_argument(
        "--classification-experiment-name",
        default=DEFAULT_CLASSIFICATION_EXPERIMENT,
        help="Output folder name for classified GeoJSONs.",
    )
    parser.add_argument(
        "--segmentation-experiment-name",
        default=DEFAULT_SEGMENTATION_EXPERIMENT,
        help="Output folder name for segmentation results.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Output folder name for the density dataset.",
    )
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=8,
        help="UNI2-h inference batch size used to reduce CUDA OOM risk.",
    )
    parser.add_argument(
        "--density-min-intersection",
        type=float,
        default=0.0,
        help="Smoke-test patch overlap threshold.",
    )
    parser.add_argument(
        "--density-min-roi-patch-area-ratio",
        type=float,
        default=0.0,
        help="Smoke-test ROI area threshold.",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Do not call setup_data.py; require local assets to already exist.",
    )
    parser.add_argument(
        "--force-downloads",
        action="store_true",
        help="Pass --force to setup_data.py downloads.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip stages whose expected smoke outputs already exist.",
    )
    parser.add_argument(
        "--allow-canonical-data-root",
        action="store_true",
        help="Allow --data-root data. By default the smoke test refuses it.",
    )
    parser.add_argument(
        "--keep-data-root",
        action="store_true",
        help="Keep the smoke-test data root after a successful run.",
    )
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def build_env(data_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["NEURO_BRAIN_FEATURE_ENCODER_DIR"] = str(
        data_root / "models" / "classification" / "feature_encoder"
    )
    return env


def require_safe_data_root(data_root: Path, allow_canonical: bool) -> None:
    if data_root.resolve() == Path("data").resolve() and not allow_canonical:
        raise SystemExit(
            "Refusing to run smoke test against data/. Use data_temp or pass "
            "--allow-canonical-data-root explicitly."
        )


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_derived_outputs(args: argparse.Namespace) -> None:
    data_root = args.data_root
    for image_id in args.image_ids:
        for path in [
            data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_HR_crop.tif",
            data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_bbox_hr.json",
            data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_contours_hr.geojson",
            data_root / "input" / "all_regions" / "low_res" / f"{image_id}_LR_crop.png",
            data_root
            / "input"
            / "all_regions"
            / "low_res"
            / f"{image_id}_bbox_lr.json",
            data_root
            / "input"
            / "all_regions"
            / "low_res"
            / f"{image_id}_contours_lr.geojson",
            data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_HR_crop.tif",
            data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_bbox_hr.json",
            data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_contours_hr.geojson",
        ]:
            remove_path(path)

    remove_path(segmentation_dir(args))
    remove_path(classification_dir(args))
    remove_path(density_dataset_dir(args))


def require_files(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing {label} outputs:\n{missing_text}")


def all_files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def raw_asset_paths(args: argparse.Namespace) -> list[Path]:
    paths = []
    for image_id in args.image_ids:
        paths.extend(
            [
                args.data_root / "raw" / "high_res" / f"B20_{image_id}.tif",
                args.data_root / "raw" / "high_res" / f"B20_{image_id}_affine.json",
                args.data_root / "raw" / "low_res" / f"pm{image_id}o.mnc",
            ]
        )
    paths.extend(
        [
            *[
                args.data_root
                / "raw"
                / "masks"
                / "3dVolumes_SegmentationMasks_40um"
                / f"sub-bbhist_hemi-R_{region[1:]}.surf.gii"
                for region in CA_REGIONS
            ],
            args.data_root
            / "models"
            / "classification"
            / "ml_classifier"
            / "model.joblib",
            args.data_root
            / "models"
            / "classification"
            / "ml_classifier"
            / "metadata.json",
            args.data_root
            / "models"
            / "segmentation"
            / "cellpose"
            / "finetune_v4_astrocytes_big_brain",
            args.data_root
            / "models"
            / "segmentation"
            / "hovernet"
            / "net_epoch=20.tar",
            args.data_root / "models" / "segmentation" / "instanseg" / "instanseg.pt",
            args.data_root / "models" / "segmentation" / "stardist" / "weights_best.h5",
        ]
    )
    return paths


def hr_preprocessing_outputs(args: argparse.Namespace) -> list[Path]:
    return [
        path
        for image_id in args.image_ids
        for path in [
            args.data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_HR_crop.tif",
            args.data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_bbox_hr.json",
            args.data_root
            / "input"
            / "all_regions"
            / "high_res"
            / f"{image_id}_contours_hr.geojson",
        ]
    ]


def lr_preprocessing_outputs(args: argparse.Namespace, image_id: str) -> list[Path]:
    return [
        args.data_root
        / "input"
        / "all_regions"
        / "low_res"
        / f"{image_id}_LR_crop.png",
        args.data_root
        / "input"
        / "all_regions"
        / "low_res"
        / f"{image_id}_bbox_lr.json",
        args.data_root
        / "input"
        / "all_regions"
        / "low_res"
        / f"{image_id}_contours_lr.geojson",
    ]


def region_preprocessing_outputs(args: argparse.Namespace) -> list[Path]:
    return [
        path
        for image_id in args.image_ids
        for path in [
            args.data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_HR_crop.tif",
            args.data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_bbox_hr.json",
            args.data_root
            / "input"
            / "single_regions"
            / "high_res"
            / args.region
            / f"{image_id}_contours_hr.geojson",
        ]
    ]


def segmentation_dir(args: argparse.Namespace) -> Path:
    return (
        args.data_root
        / "output"
        / "segmentation"
        / args.region
        / args.segmentation_experiment_name
    )


def segmentation_outputs(args: argparse.Namespace) -> list[Path]:
    return [
        segmentation_dir(args) / f"{image_id}_HR_crop_merged.geojson"
        for image_id in args.image_ids
    ]


def classification_dir(args: argparse.Namespace) -> Path:
    return (
        args.data_root
        / "output"
        / "classification"
        / args.region
        / args.classification_experiment_name
    )


def classification_outputs(args: argparse.Namespace) -> list[Path]:
    return [
        classification_dir(args) / f"{image_id}_classification_results.geojson"
        for image_id in args.image_ids
    ]


def density_dataset_dir(args: argparse.Namespace) -> Path:
    return args.data_root / "output" / "lr_density_dataset" / args.dataset_name


def validate_density_dataset(args: argparse.Namespace) -> None:
    dataset_dir = density_dataset_dir(args)
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing density dataset metadata: {info_path}")

    with info_path.open() as handle:
        dataset_info = json.load(handle)

    stats = dataset_info.get("stats", {})
    if stats.get("failed"):
        raise RuntimeError(f"Density dataset failures: {stats['failed']}")
    for split in ["train", "test"]:
        if int(stats.get(split, 0)) <= 0:
            raise RuntimeError(f"Density dataset has no {split} patches.")
        for subfolder in ["images", "densities", "roi_masks"]:
            files = list((dataset_dir / split / subfolder).iterdir())
            if not files:
                raise RuntimeError(f"No files written under {split}/{subfolder}.")


def maybe_run(
    args: argparse.Namespace,
    label: str,
    expected_outputs: list[Path],
    command: list[str],
    env: dict[str, str],
) -> None:
    if args.reuse_existing and all_files_exist(expected_outputs):
        print(f"Skipping {label}; expected outputs already exist.", flush=True)
        return
    run_command(command, env)
    require_files(expected_outputs, label)


def run_setup(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.skip_downloads:
        require_files(raw_asset_paths(args), "downloaded raw/model")
        return

    command = [
        sys.executable,
        "scripts/setup_data.py",
        "--data-root",
        str(args.data_root),
        "--download-surfaces",
        "--download-weights",
        "--download-hr",
        "--download-lr",
        "--image-ids",
        *args.image_ids,
    ]
    if args.force_downloads:
        command.insert(2, "--force")
    run_command(command, env)
    require_files(raw_asset_paths(args), "downloaded raw/model")


def run_preprocessing(args: argparse.Namespace, env: dict[str, str]) -> None:
    maybe_run(
        args,
        "HR preprocessing",
        hr_preprocessing_outputs(args),
        [
            sys.executable,
            "-m",
            "src.preprocessing.extract_crops_and_coords_HR",
            "--hr-folder-path",
            str(args.data_root / "raw" / "high_res"),
            "--outpath",
            str(args.data_root / "input" / "all_regions" / "high_res"),
            "--surfaces-folder",
            str(args.data_root / "raw" / "masks" / "3dVolumes_SegmentationMasks_40um"),
            "--mask-names",
            *CA_REGIONS,
        ],
        env,
    )

    for image_id in args.image_ids:
        maybe_run(
            args,
            f"LR preprocessing {image_id}",
            lr_preprocessing_outputs(args, image_id),
            [
                sys.executable,
                "-m",
                "src.preprocessing.extract_crops_and_coords_LR",
                "--lr-folder-path",
                str(args.data_root / "raw" / "low_res"),
                "--outpath",
                str(args.data_root / "input" / "all_regions" / "low_res"),
                "--surfaces-folder",
                str(
                    args.data_root
                    / "raw"
                    / "masks"
                    / "3dVolumes_SegmentationMasks_40um"
                ),
                "--start-idx",
                image_id,
                "--end-idx",
                image_id,
                "--num-workers",
                "1",
                "--mask-names",
                *CA_REGIONS,
            ],
            env,
        )

    maybe_run(
        args,
        f"{args.region} HR region extraction",
        region_preprocessing_outputs(args),
        [
            sys.executable,
            "-m",
            "src.preprocessing.extract_hr_region_crops",
            "--ann-dir",
            str(args.data_root / "input" / "all_regions" / "high_res"),
            "--hr-dir",
            str(args.data_root / "raw" / "high_res"),
            "--regions",
            args.region,
            "--out-dir",
            str(args.data_root / "input" / "single_regions" / "high_res"),
        ],
        env,
    )


def run_segmentation(args: argparse.Namespace, env: dict[str, str]) -> None:
    config_path = (
        Path("experiments")
        / "segmentation"
        / "allmodels"
        / (f"allmodels-{args.region}.yaml")
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing segmentation config: {config_path}")

    maybe_run(
        args,
        "segmentation",
        segmentation_outputs(args),
        [
            sys.executable,
            "-m",
            "src.segmentation.multimodel_inference",
            "--config",
            str(config_path),
            "--input_dir",
            str(args.data_root / "input" / "single_regions" / "high_res" / args.region),
            "--input_masks_dir",
            str(args.data_root / "input" / "single_regions" / "high_res" / args.region),
            "--output_dir",
            str(segmentation_dir(args)),
            "--cp_model_path",
            str(
                args.data_root
                / "models"
                / "segmentation"
                / "cellpose"
                / "finetune_v4_astrocytes_big_brain"
            ),
            "--sd_model_path",
            str(args.data_root / "models" / "segmentation" / "stardist"),
            "--hn_model_path",
            str(
                args.data_root
                / "models"
                / "segmentation"
                / "hovernet"
                / "net_epoch=20.tar"
            ),
            "--is_model_path",
            str(
                args.data_root
                / "models"
                / "segmentation"
                / "instanseg"
                / "instanseg.pt"
            ),
            "--cp_batch_size",
            "1",
            "--hn_batch_size",
            "1",
            "--is_batch_size",
            "1",
            "--sd_block_size",
            "512",
        ],
        env,
    )


def run_classification(args: argparse.Namespace, env: dict[str, str]) -> None:
    maybe_run(
        args,
        "classification",
        classification_outputs(args),
        [
            sys.executable,
            "src/classification/inference.py",
            "--model_folder",
            str(args.data_root / "models" / "classification" / "ml_classifier"),
            "--annotations_folder",
            str(segmentation_dir(args)),
            "--images_folder",
            str(args.data_root / "input" / "single_regions" / "high_res" / args.region),
            "--output_folder",
            str(classification_dir(args)),
            "--batch_size",
            str(args.classification_batch_size),
        ],
        env,
    )


def run_density_creation(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.reuse_existing:
        try:
            validate_density_dataset(args)
            print("Skipping density creation; expected outputs already exist.")
            return
        except (FileNotFoundError, RuntimeError):
            pass

    run_command(
        [
            sys.executable,
            "-m",
            "src.density_estimator.datasets.create_dataset",
            "--input-hr-dir",
            str(args.data_root / "input" / "single_regions" / "high_res"),
            "--input-hr-coords",
            str(args.data_root / "input" / "single_regions" / "high_res"),
            "--input-masks-dir",
            str(args.data_root / "output" / "classification"),
            "--classification-experiment-name",
            args.classification_experiment_name,
            "--full-hr-path",
            str(args.data_root / "raw" / "high_res"),
            "--full-lr-path",
            str(args.data_root / "raw" / "low_res"),
            "--output-dir",
            str(density_dataset_dir(args)),
            "--regions",
            args.region,
            "--test-size",
            "0.5",
            "--min-intersection",
            str(args.density_min_intersection),
            "--min-roi-patch-area-ratio",
            str(args.density_min_roi_patch_area_ratio),
        ],
        env,
    )
    validate_density_dataset(args)


def cleanup_data_root(args: argparse.Namespace) -> None:
    if args.keep_data_root:
        print(f"Keeping smoke-test data root: {args.data_root}")
        return

    if args.data_root.resolve() != Path("data_temp").resolve():
        print(
            "Keeping custom data root. Only the default data_temp root is deleted "
            "automatically."
        )
        return

    print(f"Deleting smoke-test data root: {args.data_root}")
    shutil.rmtree(args.data_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    args.image_ids = [str(image_id).zfill(4) for image_id in args.image_ids]
    if len(set(args.image_ids)) < 2:
        raise SystemExit("At least two distinct image IDs are required.")

    require_safe_data_root(args.data_root, args.allow_canonical_data_root)
    env = build_env(args.data_root)

    if not args.reuse_existing:
        clean_derived_outputs(args)

    run_setup(args, env)
    run_preprocessing(args, env)
    run_segmentation(args, env)
    run_classification(args, env)
    run_density_creation(args, env)

    print("\nTwo-WSI smoke test completed successfully.")
    print(f"Density dataset: {density_dataset_dir(args)}")
    cleanup_data_root(args)


if __name__ == "__main__":
    main()
