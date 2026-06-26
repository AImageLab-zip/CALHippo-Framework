from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

LR_BASE_URL = (
    "https://ftp.bigbrainproject.org/bigbrain-ftp/"
    "BigBrainRelease.2015/2D_Final_Sections/Coronal/Minc/"
)
HR_ALIGNED_BASE_URL = (
    "https://data-proxy.ebrains.eu/api/v1/buckets/"
    "p22717-hbp-d000070_BigBrain-selected_1um_scans_pub/v1.0/aligned/"
)
SURFACE_BASE_URL = (
    "https://ftp.bigbrainproject.org/bigbrain-ftp/"
    "BigBrainRelease.2015/Hippocampus_Segmentation/gii/"
)

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 120)
DEFAULT_IDS_FILE = Path(__file__).with_name("default_lr_ids.txt")
CALHIPPO_DATASET_SHA256 = (
    "1ee534f851471696a6d418e08b7dd7968e0a9bdf2fcd0e2a62e746577ce78754"
)
CALHIPPO_DATASET_ROOT_NAME = "CALHippo_Dataset_v1.0"
CALHIPPO_DATASET_POINT_CLOUD_RUN = "calhippo_dataset_v1.0"
DEFAULT_CLASSIFICATION_EXPERIMENT = "ml_classifier_logistic_encoder_uni2h"
HF_MODELS_REPO_ID = "AImageLab-Zip/CALHippo-Framework-Models"
HF_MODELS_REVISION = "main"
REGIONS = ["RCA1", "RCA2", "RCA3", "RCA4"]
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
SEGMENTATION_MODEL_FILES = [
    "segmentation/cellpose/finetune_v4_astrocytes_big_brain",
    "segmentation/hovernet/net_epoch=20.tar",
    "segmentation/instanseg/instanseg.pt",
    "segmentation/stardist/config.json",
    "segmentation/stardist/thresholds.json",
    "segmentation/stardist/weights_best.h5",
]
CLASSIFICATION_MODEL_DIR = Path("classification") / "ml_classifier"
CLASSIFICATION_FEATURE_ENCODER_DIR = Path("classification") / "feature_encoder"
DENSITY_MODEL_DIR = Path("density_estimation") / "short_unet"
DENSITY_MODEL_FILE = str(DENSITY_MODEL_DIR / "final_density_model.pth")
DENSITY_CONFIG_FILE = str(
    DENSITY_MODEL_DIR
    / "9_shorter_unet_normalizedgame_asymclassnormalizedl1loss_adamw.yaml"
)

CLASSIFICATION_WEIGHT_FILES = [
    str(CLASSIFICATION_MODEL_DIR / "model.joblib"),
    str(CLASSIFICATION_MODEL_DIR / "metadata.json"),
    str(CLASSIFICATION_MODEL_DIR / "metrics.json"),
]

HF_WEIGHT_FILES = [
    DENSITY_MODEL_FILE,
    DENSITY_CONFIG_FILE,
    *SEGMENTATION_MODEL_FILES,
    *CLASSIFICATION_WEIGHT_FILES,
]

SURFACE_FILES = [
    "sub-bbhist_hemi-L_CA1.surf.gii",
    "sub-bbhist_hemi-L_CA2.surf.gii",
    "sub-bbhist_hemi-L_CA3.surf.gii",
    "sub-bbhist_hemi-L_CA4.surf.gii",
    "sub-bbhist_hemi-L_DG.surf.gii",
    "sub-bbhist_hemi-L_midSurf.surf.gii",
    "sub-bbhist_hemi-L_Sub.surf.gii",
    "sub-bbhist_hemi-R_CA1.surf.gii",
    "sub-bbhist_hemi-R_CA2.surf.gii",
    "sub-bbhist_hemi-R_CA3.surf.gii",
    "sub-bbhist_hemi-R_CA4.surf.gii",
    "sub-bbhist_hemi-R_DG.surf.gii",
    "sub-bbhist_hemi-R_midSurf.surf.gii",
    "sub-bbhist_hemi-R_Sub.surf.gii",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the neuro_brain data tree and optionally download public "
            "BigBrain assets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data folder to create and populate.",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help=(
            "Download the maintained public setup in one command: surfaces, "
            "default HR/LR IDs, and model weights."
        ),
    )
    parser.add_argument(
        "--download-surfaces",
        action="store_true",
        help="Download hippocampal .surf.gii files from the BigBrain FTP.",
    )
    parser.add_argument(
        "--download-lr",
        action="store_true",
        help=(
            "Download LR coronal .mnc files. Defaults to the maintained full "
            f"range from {DEFAULT_IDS_FILE.name} when no IDs are provided."
        ),
    )
    parser.add_argument(
        "--image-ids",
        nargs="*",
        default=[],
        help="Image IDs to download for HR/LR data, e.g. 0047 0102 3196.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "Text file with one image ID or inclusive ID range per line. "
            f"Defaults to {DEFAULT_IDS_FILE} when no IDs are provided."
        ),
    )
    parser.add_argument(
        "--lr-range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive image ID range for HR/LR data, e.g. 2777 3998.",
    )
    parser.add_argument(
        "--download-hr",
        action="store_true",
        help=(
            "Download aligned HR .tif files and matching affine .json files. "
            f"Defaults to the IDs from {DEFAULT_IDS_FILE.name} when no IDs are "
            "provided."
        ),
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download released model artifacts from Hugging Face.",
    )
    parser.add_argument(
        "--calhippo-dataset-zip",
        type=Path,
        default=None,
        help=(
            "Path to CALHippo_Dataset_v1.0.zip downloaded from the CALHippo "
            "dataset website. Verifies SHA-256, extracts it, places released "
            "HR crops/annotations/point cloud into the data tree, and downloads "
            "the matching HR affine JSONs."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Folder where Hugging Face model artifacts are downloaded.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=HF_MODELS_REPO_ID,
        help="Hugging Face model repository ID for model artifacts.",
    )
    parser.add_argument(
        "--hf-revision",
        default=HF_MODELS_REVISION,
        help="Hugging Face repository revision for model artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing downloaded files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without creating folders or downloading files.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Check that selected remote assets are reachable without writing "
            "files or downloading payloads."
        ),
    )
    return parser.parse_args()


def required_dirs(data_root: Path) -> list[Path]:
    dirs = [
        data_root / "raw" / "high_res",
        data_root / "raw" / "low_res",
        data_root / "raw" / "masks" / "3dVolumes_SegmentationMasks_40um",
        data_root / "input" / "all_regions" / "high_res",
        data_root / "input" / "all_regions" / "low_res",
        data_root / "input" / "train_test_splits" / "segmentation",
        data_root / "input" / "train_test_splits" / "classification",
        data_root / "input" / "custom_masks",
        data_root / "input" / "classification_gt",
        data_root / "misc",
        data_root / "output" / "lr_density_dataset",
        data_root / "output" / "test_lr_density_gt",
        data_root / "output" / "lr_gt_eval",
        data_root / "output" / "full_lr_predictions",
        data_root / "output" / "mesoscale_reconstruction",
        data_root / "density_estimator_training",
        data_root / "models" / "classification" / "ml_classifier",
        data_root / "models" / "classification" / "feature_encoder",
        data_root / "models" / "density_estimation" / "short_unet",
        data_root / "models" / "segmentation" / "cellpose",
        data_root / "models" / "segmentation" / "hovernet",
        data_root / "models" / "segmentation" / "instanseg",
        data_root / "models" / "segmentation" / "stardist",
        data_root / "models" / "original_weights" / "classification",
        data_root / "models" / "original_weights" / "density_estimation",
        data_root / "models" / "original_weights" / "segmentation",
    ]
    for region in REGIONS:
        dirs.extend(
            [
                data_root / "input" / "single_regions" / "high_res" / region,
                data_root / "output" / "segmentation" / region,
                data_root / "output" / "classification" / region,
            ]
        )
    return dirs


def create_dirs(paths: list[Path], dry_run: bool) -> None:
    for path in paths:
        if dry_run:
            print(f"DRY-RUN create directory: {path}")
            continue
        path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory: {path}")


def normalize_image_id(value: str) -> str:
    match = re.search(r"(\d+)", Path(value).name)
    if match is None:
        raise ValueError(f"Could not parse an image ID from: {value}")
    return match.group(1).zfill(4)


def expand_id_range(start: str, end: str) -> list[str]:
    start_id = int(normalize_image_id(start))
    end_id = int(normalize_image_id(end))
    if start_id > end_id:
        raise ValueError("ID range START must be <= END")
    return [str(image_id).zfill(4) for image_id in range(start_id, end_id + 1)]


def read_ids_file(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", line)
        if range_match:
            ids.extend(expand_id_range(range_match.group(1), range_match.group(2)))
            continue
        ids.append(normalize_image_id(line))
    return ids


def collect_image_ids(args: argparse.Namespace) -> list[str]:
    ids = []
    has_explicit_ids = bool(args.image_ids or args.ids_file or args.lr_range)

    ids.extend(normalize_image_id(value) for value in args.image_ids)

    if args.ids_file is not None:
        ids.extend(read_ids_file(args.ids_file))

    if args.lr_range is not None:
        ids.extend(expand_id_range(args.lr_range[0], args.lr_range[1]))

    if not has_explicit_ids:
        if not DEFAULT_IDS_FILE.exists():
            raise SystemExit(
                "ERROR: no image IDs were provided and the default ID file is "
                f"missing: {DEFAULT_IDS_FILE}"
            )
        ids.extend(read_ids_file(DEFAULT_IDS_FILE))

    if not ids:
        raise SystemExit("ERROR: no image IDs resolved for download.")

    return sorted(set(ids), key=int)


def build_download_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(
    url: str,
    output_path: Path,
    force: bool,
    dry_run: bool,
    description: str | None = None,
    session: requests.Session | None = None,
) -> None:
    if output_path.exists() and not force:
        print(f"Exists, skipping: {output_path}")
        return

    if dry_run:
        print(f"DRY-RUN download: {url} -> {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    if partial_path.exists():
        partial_path.unlink()

    print(f"Downloading: {url}")
    close_session = session is None
    session = session or build_download_session()
    try:
        with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            total = int(content_length) if content_length else None

            with partial_path.open("wb") as handle:
                progress = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=description or output_path.name,
                )
                with progress:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))

        if total is not None and partial_path.stat().st_size != total:
            raise RuntimeError(
                f"Downloaded size mismatch for {output_path}: "
                f"expected {total} bytes, got {partial_path.stat().st_size}."
            )
        partial_path.replace(output_path)
    except (OSError, requests.RequestException) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    finally:
        if close_session:
            session.close()
    print(f"Saved: {output_path}")


def remote_file_exists(session: requests.Session, url: str) -> bool:
    try:
        with session.head(
            url,
            allow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
        ) as response:
            if response.status_code in {200, 206}:
                return True
            if response.status_code == 404:
                return False

        with session.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
        ) as fallback_response:
            if fallback_response.status_code in {200, 206}:
                return True
            if fallback_response.status_code == 404:
                return False
            fallback_response.raise_for_status()
            return True
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to check remote file: {url}") from exc


def check_remote_file(session: requests.Session, url: str, description: str) -> bool:
    exists = remote_file_exists(session, url)
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {description}: {url}")
    return exists


def hr_image_url(image_id: str) -> str:
    return f"{urljoin(HR_ALIGNED_BASE_URL, f'B20_{image_id}.tif')}?inline=true"


def hr_affine_url(image_id: str) -> str:
    return f"{urljoin(HR_ALIGNED_BASE_URL, f'B20_{image_id}_affine.json')}?inline=true"


def download_hr_images(
    data_root: Path,
    image_ids: list[str],
    force: bool,
    dry_run: bool,
) -> None:
    output_dir = data_root / "raw" / "high_res"

    if dry_run:
        for image_id in image_ids:
            print(
                "DRY-RUN download HR pair: "
                f"{hr_image_url(image_id)} -> {output_dir / f'B20_{image_id}.tif'}"
            )
            print(
                "DRY-RUN download HR affine: "
                f"{hr_affine_url(image_id)} -> "
                f"{output_dir / f'B20_{image_id}_affine.json'}"
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with build_download_session() as session:
        for image_id in image_ids:
            image_path = output_dir / f"B20_{image_id}.tif"
            affine_path = output_dir / f"B20_{image_id}_affine.json"
            needs_download = (
                force or not image_path.exists() or not affine_path.exists()
            )

            if needs_download:
                image_url = hr_image_url(image_id)
                affine_url = hr_affine_url(image_id)
                if not remote_file_exists(session, image_url):
                    print(
                        f"Missing HR TIFF upstream, skipping ID {image_id}: {image_url}"
                    )
                    continue
                if not remote_file_exists(session, affine_url):
                    print(
                        f"Missing HR affine upstream, skipping ID {image_id}: "
                        f"{affine_url}"
                    )
                    continue

                download_file(
                    url=image_url,
                    output_path=image_path,
                    force=force,
                    dry_run=dry_run,
                    description=f"B20_{image_id}.tif",
                    session=session,
                )
                download_file(
                    url=affine_url,
                    output_path=affine_path,
                    force=force,
                    dry_run=dry_run,
                    description=f"B20_{image_id}_affine.json",
                    session=session,
                )

    print(
        f"HR download complete for {len(image_ids)} requested IDs. "
        "Missing upstream IDs were skipped."
    )


def download_hr_affines(
    data_root: Path,
    image_ids: list[str],
    force: bool,
    dry_run: bool,
) -> None:
    output_dir = data_root / "raw" / "high_res"
    with build_download_session() as session:
        for image_id in image_ids:
            download_file(
                url=hr_affine_url(image_id),
                output_path=output_dir / f"B20_{image_id}_affine.json",
                force=force,
                dry_run=dry_run,
                description=f"B20_{image_id}_affine.json",
                session=session,
            )


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(zip_path: Path, output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN extract: {zip_path} -> {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = (output_dir / member.filename).resolve()
            if output_root not in target_path.parents and target_path != output_root:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(output_dir)


def copy_path(src: Path, dst: Path, force: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN copy: {src} -> {dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"Expected dataset path not found: {src}")
    if dst.exists() and force:
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ingest_calhippo_dataset_zip(
    zip_path: Path,
    data_root: Path,
    force: bool,
    dry_run: bool,
) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"CALHippo dataset zip not found: {zip_path}")

    print(f"Verifying SHA-256: {zip_path}")
    actual_sha256 = sha256sum(zip_path)
    if actual_sha256 != CALHIPPO_DATASET_SHA256:
        raise RuntimeError(
            "CALHippo dataset checksum mismatch. "
            f"Expected {CALHIPPO_DATASET_SHA256}, got {actual_sha256}."
        )
    print("Checksum OK.")

    extract_dir = data_root / "misc" / "calhippo_dataset_release"
    dataset_dir = extract_dir / CALHIPPO_DATASET_ROOT_NAME
    safe_extract_zip(zip_path, extract_dir, dry_run=dry_run)

    hr_images_dir = dataset_dir / "HR_annotations" / "HR_images"
    annotations_dir = dataset_dir / "HR_annotations" / "HR_annotations"
    point_cloud_path = dataset_dir / "point_cloud" / "point_cloud.csv"

    if not dry_run:
        for expected_path in [hr_images_dir, annotations_dir, point_cloud_path]:
            if not expected_path.exists():
                raise FileNotFoundError(
                    f"Expected CALHippo dataset path not found: {expected_path}"
                )

    for region in REGIONS:
        copy_path(
            src=hr_images_dir / region,
            dst=data_root / "input" / "single_regions" / "high_res" / region,
            force=force,
            dry_run=dry_run,
        )
        output_dir = (
            data_root
            / "output"
            / "classification"
            / region
            / DEFAULT_CLASSIFICATION_EXPERIMENT
        )
        if dry_run:
            print(f"DRY-RUN create directory: {output_dir}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        annotation_paths = sorted(
            (annotations_dir / region).glob("*_classification_results.geojson")
        )
        if not annotation_paths and not dry_run:
            raise FileNotFoundError(
                "No classification GeoJSON files found under "
                f"{annotations_dir / region}"
            )
        for annotation_path in annotation_paths:
            copy_path(
                src=annotation_path,
                dst=output_dir / annotation_path.name,
                force=force,
                dry_run=dry_run,
            )

    point_cloud_output = (
        data_root
        / "output"
        / "mesoscale_reconstruction"
        / CALHIPPO_DATASET_POINT_CLOUD_RUN
        / "point_cloud.csv"
    )
    copy_path(
        src=point_cloud_path,
        dst=point_cloud_output,
        force=force,
        dry_run=dry_run,
    )

    image_ids = sorted(
        {path.name.split("_")[0] for path in hr_images_dir.glob("RCA*/*_HR_crop.tif")},
        key=int,
    )
    if not image_ids and not dry_run:
        raise FileNotFoundError(f"No HR crop files found under {hr_images_dir}")
    download_hr_affines(
        data_root=data_root,
        image_ids=image_ids,
        force=force,
        dry_run=dry_run,
    )
    print(f"CALHippo dataset placed under data tree. Point cloud: {point_cloud_output}")


def test_hr_images_available(image_ids: list[str]) -> bool:
    all_ok = True
    with build_download_session() as session:
        for image_id in image_ids:
            image_ok = check_remote_file(
                session,
                hr_image_url(image_id),
                f"HR TIFF for {image_id}",
            )
            affine_ok = check_remote_file(
                session,
                hr_affine_url(image_id),
                f"HR affine for {image_id}",
            )
            all_ok = all_ok and image_ok and affine_ok
    return all_ok


def download_hf_file(
    repo_id: str,
    revision: str,
    file_name: str,
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> Path:
    output_path = output_dir / file_name
    if dry_run:
        print(f"DRY-RUN hf download: {repo_id}:{revision}:{file_name} -> {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Hugging Face: {repo_id}:{revision}:{file_name}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="model",
            revision=revision,
            local_dir=output_dir,
            force_download=force,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {file_name} from Hugging Face repo {repo_id}. "
            "For private repos, run `hf auth login` or set HF_TOKEN."
        ) from exc

    return Path(downloaded_path)


def download_model_weights(
    data_root: Path,
    weights_dir: Path | None,
    repo_id: str,
    revision: str,
    force: bool,
    dry_run: bool,
) -> None:
    weights_dir = weights_dir or data_root / "models"
    for file_name in HF_WEIGHT_FILES:
        download_hf_file(
            repo_id=repo_id,
            revision=revision,
            file_name=file_name,
            output_dir=weights_dir,
            force=force,
            dry_run=dry_run,
        )
    print(f"Density model folder: {weights_dir / DENSITY_MODEL_DIR}")
    print(f"Segmentation model folder: {weights_dir / 'segmentation'}")
    if CLASSIFICATION_WEIGHT_FILES:
        print(f"Classification model folder: {weights_dir / CLASSIFICATION_MODEL_DIR}")
    else:
        print(
            "Classification model folder placeholder: "
            f"{weights_dir / CLASSIFICATION_MODEL_DIR}"
        )
    print(
        "Classification feature encoder cache: "
        f"{weights_dir / CLASSIFICATION_FEATURE_ENCODER_DIR}"
    )


def download_surfaces(data_root: Path, force: bool, dry_run: bool) -> None:
    output_dir = data_root / "raw" / "masks" / "3dVolumes_SegmentationMasks_40um"
    for file_name in SURFACE_FILES:
        download_file(
            url=urljoin(SURFACE_BASE_URL, file_name),
            output_path=output_dir / file_name,
            force=force,
            dry_run=dry_run,
        )


def test_surfaces_available() -> bool:
    all_ok = True
    with build_download_session() as session:
        for file_name in SURFACE_FILES:
            all_ok = (
                check_remote_file(
                    session,
                    urljoin(SURFACE_BASE_URL, file_name),
                    f"Surface {file_name}",
                )
                and all_ok
            )
    return all_ok


def download_lr_images(
    data_root: Path,
    image_ids: list[str],
    force: bool,
    dry_run: bool,
) -> None:
    output_dir = data_root / "raw" / "low_res"
    with build_download_session() as session:
        for image_id in image_ids:
            file_name = f"pm{image_id}o.mnc"
            download_file(
                url=urljoin(LR_BASE_URL, file_name),
                output_path=output_dir / file_name,
                force=force,
                dry_run=dry_run,
                session=session,
            )


def test_lr_images_available(image_ids: list[str]) -> bool:
    all_ok = True
    with build_download_session() as session:
        for image_id in image_ids:
            file_name = f"pm{image_id}o.mnc"
            all_ok = (
                check_remote_file(
                    session,
                    urljoin(LR_BASE_URL, file_name),
                    f"LR MNC for {image_id}",
                )
                and all_ok
            )
    return all_ok


def test_model_weights_available(repo_id: str, revision: str) -> bool:
    all_ok = True
    for file_name in HF_WEIGHT_FILES:
        url = hf_hub_url(
            repo_id=repo_id,
            filename=file_name,
            repo_type="model",
            revision=revision,
        )
        try:
            get_hf_file_metadata(url)
            print(f"[OK] Hugging Face file: {repo_id}:{revision}:{file_name}")
        except Exception:
            print(f"[MISSING] Hugging Face file: {repo_id}:{revision}:{file_name}")
            all_ok = False
    return all_ok


def test_calhippo_dataset_zip(zip_path: Path) -> bool:
    if not zip_path.exists():
        print(f"[MISSING] CALHippo dataset zip: {zip_path}")
        return False
    actual_sha256 = sha256sum(zip_path)
    if actual_sha256 != CALHIPPO_DATASET_SHA256:
        print(
            "[MISMATCH] CALHippo dataset zip checksum: "
            f"expected {CALHIPPO_DATASET_SHA256}, got {actual_sha256}"
        )
        return False
    print(f"[OK] CALHippo dataset zip checksum: {zip_path}")
    return True


def print_next_steps(data_root: Path) -> None:
    print("\nData setup complete.")
    print(f"Data root: {data_root.resolve() if data_root.exists() else data_root}")
    print("\nNext steps:")
    print("1. Read documents/data_setup.md for the expected raw and derived files.")
    print(
        "2. Read documents/pipeline.md for the linear raw-data-to-point-cloud commands."
    )
    print(
        "3. If data_root is not ./data, pass explicit paths where scripts support them."
    )
    print("   TODO: some configs still assume repo-relative data/ paths.")


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    download_surfaces_flag = args.download_all or args.download_surfaces
    download_hr_flag = args.download_all or args.download_hr
    download_lr_flag = args.download_all or args.download_lr
    download_weights_flag = args.download_all or args.download_weights
    calhippo_dataset_flag = args.calhippo_dataset_zip is not None
    image_ids = collect_image_ids(args) if download_hr_flag or download_lr_flag else []

    create_dirs(required_dirs(data_root), dry_run=args.dry_run or args.test)

    if args.test:
        all_ok = True

        if download_surfaces_flag:
            all_ok = test_surfaces_available() and all_ok

        if download_hr_flag:
            all_ok = test_hr_images_available(image_ids) and all_ok

        if download_lr_flag:
            all_ok = test_lr_images_available(image_ids) and all_ok

        if download_weights_flag:
            all_ok = (
                test_model_weights_available(args.hf_repo_id, args.hf_revision)
                and all_ok
            )

        if calhippo_dataset_flag:
            all_ok = test_calhippo_dataset_zip(args.calhippo_dataset_zip) and all_ok

        if all_ok:
            print("\nTest mode passed: all selected assets are reachable.")
            return
        raise SystemExit("\nTest mode failed: one or more selected assets are missing.")

    if download_surfaces_flag:
        download_surfaces(data_root=data_root, force=args.force, dry_run=args.dry_run)

    if download_hr_flag:
        download_hr_images(
            data_root=data_root,
            image_ids=image_ids,
            force=args.force,
            dry_run=args.dry_run,
        )

    if download_lr_flag:
        download_lr_images(
            data_root=data_root,
            image_ids=image_ids,
            force=args.force,
            dry_run=args.dry_run,
        )

    if download_weights_flag:
        download_model_weights(
            data_root=data_root,
            weights_dir=args.weights_dir,
            repo_id=args.hf_repo_id,
            revision=args.hf_revision,
            force=args.force,
            dry_run=args.dry_run,
        )

    if calhippo_dataset_flag:
        ingest_calhippo_dataset_zip(
            zip_path=args.calhippo_dataset_zip,
            data_root=data_root,
            force=args.force,
            dry_run=args.dry_run,
        )

    print_next_steps(data_root=data_root)


if __name__ == "__main__":
    main()
