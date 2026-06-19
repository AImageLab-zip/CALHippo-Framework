import argparse
import os
import pprint
from typing import List

import yaml
from loguru import logger

from src.segmentation.model_runs import (
    AdaptiveThresholdModelRun,
    BaseModelRun,
    CellposeModelRun,
    HoverNetModelRun,
    InstanSegModelRun,
    StarDistModelRun,
)

# ==========================================
# 1. HARDCODED DEFAULTS (Fallback constants)
# ==========================================

# IO / System
DEFAULT_INPUT_DIR = "data/input/single_regions/high_res/RCA3"
DEFAULT_INPUT_MASKS_DIR = "data/input/single_regions/high_res/RCA3"
DEFAULT_OUTPUT_DIR = "data/output/segmentation/RCA3/default"
DEFAULT_IMG_EXT = ".tif"
DEFAULT_MASK_EXTS = [".geojson"]
DEFAULT_PADDING = 0
DEFAULT_SAVE_INTERMEDIATE = False
DEFAULT_LOAD_INTERMEDIATE_DIR = None
DEFAULT_DEBUG = False
DEFAULT_USE_WANDB = True
DEFAULT_WANDB_PROJECT = "neuro_brain_project"
DEFAULT_WANDB_GROUP = "inference"

# Cellpose
DEFAULT_CP_MODEL_PATH = (
    "data/models/segmentation/cellpose/finetune_v4_astrocytes_big_brain"
)
DEFAULT_CP_BATCH_SIZE = None  # None usually implies auto or 8 in many libs
DEFAULT_CP_FLOW_THRESHOLD = 0.6
DEFAULT_CP_CELLPROB_THRESHOLD = 0.0
DEFAULT_CP_DIAMETERS = [
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]

# Stardist
DEFAULT_SD_MODEL_PATH = "data/models/segmentation/stardist"
DEFAULT_SD_BLOCK_SIZE = 1024
DEFAULT_SD_PROB_THRESHOLD = 0.5
DEFAULT_SD_MAX_AREA = 10000
DEFAULT_EXCLUDE_STARDIST = False

# HoverNet
DEFAULT_HN_MODEL_PATH = "data/models/segmentation/hovernet/net_epoch=20.tar"
DEFAULT_HN_MODEL_MODE = "original"  # 'original' or 'fast'
DEFAULT_HN_NR_TYPES = None  # None for binary segmentation
DEFAULT_HN_BATCH_SIZE = 32
DEFAULT_HN_MAX_AREA = 10000
DEFAULT_EXCLUDE_HOVERNET = True  # Excluded by default since model path is required

# InstanSeg
DEFAULT_IN_MODEL_PATH = "data/models/segmentation/instanseg/instanseg.pt"
DEFAULT_IN_PATCH_SIZE = 512
DEFAULT_IN_BATCH_SIZE = 4
DEFAULT_IS_MAX_AREA = 10000
DEFAULT_EXCLUDE_INSTANSEG = True  # Excluded by default since model path is required

# Merging / Post-processing
DEFAULT_MIN_AREA_THRESHOLD = 5
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MIN_VOTE_RATIO = 0.3
DEFAULT_MAX_MEAN_COLOR = 130.0
DEFAULT_SMALL_DIAM_AREA_THRESHOLD = 100
DEFAULT_PREDICT_OUTSIDE_ROIS = False

# Complex Parameters (Adaptive Threshold)
DEFAULT_EXCLUDE_ADAPTIVE = False
DEFAULT_ADAPTIVE_PARAMS = [
    {"method": "cv2", "window_size": 15, "second_param": 3, "max_eccentricity": 0.8},
    {"method": "cv2", "window_size": 27, "second_param": 5, "max_eccentricity": 0.8},
    {
        "method": "sauvola",
        "window_size": 25,
        "second_param": 0.2,
        "max_eccentricity": 0.9,
    },
    {
        "method": "sauvola",
        "window_size": 41,
        "second_param": 0.25,
        "max_eccentricity": 0.9,
    },
]


def str2bool(v):
    """Helper to convert string flags to booleans."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# -----------------------------------------
# MODEL BLUEPRINTT EXTRACTION
# -----------------------------------------


def load_models_and_configs(args) -> List[BaseModelRun]:
    """
    Parse the provided arguments and construct a
    list of model blueprints (config dicts) that will be used for inference.

    """
    model_runs = []

    # CELLPOSE
    if args.cp_diameters:
        cp_params = {
            "batch_size": args.cp_batch_size,
            "flow_threshold": args.cp_flow_threshold,
            "cellprob_threshold": args.cp_cellprob_threshold,
            "normalize": True,
            # TODO: this enables TTA and maybe it's more accurate but slower.
            "augment": False,
        }

        for d in args.cp_diameters:
            model_runs.append(
                CellposeModelRun(
                    model_path=args.cp_model_path,
                    params={**cp_params, "diameter": d},
                )
            )

        logger.info(
            f"Registered Cellpose blueprints for diameters: {args.cp_diameters}"
        )
    else:
        logger.warning("No Cellpose diameters provided. Skipping Cellpose models.")

    # STARDIST
    if not args.exclude_stardist:
        model_runs.append(
            StarDistModelRun(
                model_path=args.sd_model_path,
                params={
                    "axes": "YXC",
                    "prob_thresh": args.sd_prob_threshold,
                    "block_size": args.sd_block_size,
                },
            )
        )
        logger.info("Registered StarDist blueprint.")
    else:
        logger.warning("StarDist model skipped in configuration.")

    # HOVERNET
    if not args.exclude_hovernet and args.hn_model_path is not None:
        model_runs.append(
            HoverNetModelRun(
                model_path=args.hn_model_path,
                params={
                    "model_mode": args.hn_model_mode,
                    "nr_types": args.hn_nr_types
                    if args.hn_nr_types and args.hn_nr_types > 0
                    else None,
                    "batch_size": args.hn_batch_size,
                },
            )
        )
        logger.info(f"Registered HoverNet blueprint with mode={args.hn_model_mode}.")
    elif not args.exclude_hovernet and args.hn_model_path is None:
        logger.warning("HoverNet enabled but no model path provided. Skipping.")
    else:
        logger.warning("HoverNet model skipped in configuration.")

    # INSTANSEG
    if not args.exclude_instanseg and args.is_model_path is not None:
        model_runs.append(
            InstanSegModelRun(
                model_path=args.is_model_path,
                params={
                    "patch_size": args.is_patch_size,
                    "batch_size": args.is_batch_size,
                    "pixel_size": 1,
                },
            )
        )
        logger.info(
            f"Registered InstanSeg blueprint with model path={args.is_model_path}."
        )

    # ADAPTIVE THRESHOLD
    # No model path needed, parameters are in the config
    if not args.exclude_adaptive_threshold:
        for at_param in args.adaptive_threshold_parameters:
            model_runs.append(AdaptiveThresholdModelRun(params=at_param))
        logger.info(
            "Registered "
            f"{len(args.adaptive_threshold_parameters)} "
            "Adaptive Threshold blueprints."
        )
    else:
        logger.warning("Adaptive Threshold models skipped in configuration.")

    return model_runs


# ---------------------------------------------------------
# CONFIG PARSING & ARGUMENT INJECTION
# ---------------------------------------------------------


def get_args():

    # First, parse only the --config argument to locate the YAML file
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", required=True, help="Path to YAML config")
    known_args, remaining_argv = conf_parser.parse_known_args()

    config_path = known_args.config
    config = {}

    if os.path.exists(config_path):
        print(f"[INFO] Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"[WARN] Config file {config_path} not found. Using internal defaults.")

    # Extract Sections (Safe access with defaults handled in add_argument)
    # We use empty dicts {} here just to avoid KeyErrors, actual defaults are below
    io_conf = config.get("IO", {})
    cp_conf = config.get("CELLPOSE", {})
    sd_conf = config.get("STARDIST", {})
    hn_conf = config.get("HOVERNET", {})
    is_conf = config.get("INSTANSEG", {})
    adaptive_conf = config.get("ADAPTIVE_THRESHOLD", {})
    merge_conf = config.get("MERGING", {})

    # ---------------------------------------------------------
    # 3. SETUP MAIN ARGUMENT PARSER
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Segmentation Pipeline: Config -> Defaults -> CLI Args",
        parents=[conf_parser],  # Inherit the --config arg
    )

    # --- IO Arguments ---
    parser.add_argument(
        "--input_dir", type=str, default=io_conf.get("input_dir", DEFAULT_INPUT_DIR)
    )
    parser.add_argument(
        "--input_masks_dir",
        type=str,
        default=io_conf.get("input_masks_dir", DEFAULT_INPUT_MASKS_DIR),
    )
    parser.add_argument(
        "--output_dir", type=str, default=io_conf.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        "--img_ext", type=str, default=io_conf.get("img_ext", DEFAULT_IMG_EXT)
    )
    parser.add_argument(
        "--mask_exts", nargs="+", default=io_conf.get("mask_exts", DEFAULT_MASK_EXTS)
    )
    parser.add_argument(
        "--padding", type=int, default=io_conf.get("padding", DEFAULT_PADDING)
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=io_conf.get("save_intermediate", DEFAULT_SAVE_INTERMEDIATE),
    )
    parser.add_argument(
        "--load_intermediate_dir",
        type=str,
        default=io_conf.get("load_intermediate_dir", DEFAULT_LOAD_INTERMEDIATE_DIR),
    )
    parser.add_argument(
        "--debug", action="store_true", default=io_conf.get("debug", DEFAULT_DEBUG)
    )
    parser.add_argument(
        "--use_wandb",
        type=str2bool,
        default=io_conf.get("use_wandb", DEFAULT_USE_WANDB),
        help="Enable/disable WandB logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=io_conf.get("wandb_project", DEFAULT_WANDB_PROJECT),
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=io_conf.get("wandb_group", DEFAULT_WANDB_GROUP),
        help="WandB group name",
    )

    # --- Merging Parameters ---
    parser.add_argument(
        "--min_area_threshold",
        type=float,
        default=merge_conf.get("min_area_threshold", DEFAULT_MIN_AREA_THRESHOLD),
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=merge_conf.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
    )
    parser.add_argument(
        "--min_vote_ratio",
        type=float,
        default=merge_conf.get("min_vote_ratio", DEFAULT_MIN_VOTE_RATIO),
    )
    parser.add_argument(
        "--max_mean_color",
        type=float,
        default=merge_conf.get("max_mean_color", DEFAULT_MAX_MEAN_COLOR),
    )
    parser.add_argument(
        "--small_diam_area_threshold",
        type=float,
        default=merge_conf.get(
            "small_diam_area_threshold", DEFAULT_SMALL_DIAM_AREA_THRESHOLD
        ),
    )
    parser.add_argument(
        "--predict_outside_rois",
        action="store_true",
        default=merge_conf.get("predict_outside_rois", DEFAULT_PREDICT_OUTSIDE_ROIS),
    )

    # --- Cellpose Parameters ---
    parser.add_argument(
        "--cp_model_path",
        type=str,
        default=cp_conf.get("model_path", DEFAULT_CP_MODEL_PATH),
    )
    parser.add_argument(
        "--cp_batch_size",
        type=int,
        default=cp_conf.get("batch_size", DEFAULT_CP_BATCH_SIZE),
    )
    parser.add_argument(
        "--cp_flow_threshold",
        type=float,
        default=cp_conf.get("flow_threshold", DEFAULT_CP_FLOW_THRESHOLD),
    )
    parser.add_argument(
        "--cp_cellprob_threshold",
        type=float,
        default=cp_conf.get("cellprob_threshold", DEFAULT_CP_CELLPROB_THRESHOLD),
    )
    parser.add_argument(
        "--cp_diameters",
        nargs="+",
        type=float,
        default=cp_conf.get("diameters", DEFAULT_CP_DIAMETERS),
    )

    # --- Stardist Parameters ---
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default=sd_conf.get("model_path", DEFAULT_SD_MODEL_PATH),
    )
    parser.add_argument(
        "--sd_block_size",
        type=int,
        default=sd_conf.get("block_size", DEFAULT_SD_BLOCK_SIZE),
    )
    parser.add_argument(
        "--sd_prob_threshold",
        type=float,
        default=sd_conf.get("prob_threshold", DEFAULT_SD_PROB_THRESHOLD),
    )
    parser.add_argument(
        "--sd_max_area",
        type=float,
        default=sd_conf.get("max_area", DEFAULT_SD_MAX_AREA),
    )
    parser.add_argument(
        "--exclude_stardist",
        action="store_true",
        default=sd_conf.get("exclude_stardist", DEFAULT_EXCLUDE_STARDIST),
    )

    # --- HoverNet Parameters ---
    parser.add_argument(
        "--hn_model_path",
        type=str,
        default=hn_conf.get("model_path", DEFAULT_HN_MODEL_PATH),
    )
    parser.add_argument(
        "--hn_model_mode",
        type=str,
        default=hn_conf.get("model_mode", DEFAULT_HN_MODEL_MODE),
        choices=["original", "fast"],
    )
    parser.add_argument(
        "--hn_nr_types",
        type=int,
        default=hn_conf.get("nr_types", DEFAULT_HN_NR_TYPES),
    )
    parser.add_argument(
        "--hn_batch_size",
        type=int,
        default=hn_conf.get("batch_size", DEFAULT_HN_BATCH_SIZE),
    )
    parser.add_argument(
        "--exclude_hovernet",
        action="store_true",
        default=hn_conf.get("exclude_hovernet", DEFAULT_EXCLUDE_HOVERNET),
    )
    parser.add_argument(
        "--hn_max_area",
        type=float,
        default=hn_conf.get("max_area", DEFAULT_HN_MAX_AREA),
    )

    # --- InstanSeg Parameters ---
    parser.add_argument(
        "--is_model_path",
        type=str,
        default=is_conf.get(
            "model_path", is_conf.get("model_name", DEFAULT_IN_MODEL_PATH)
        ),
    )
    parser.add_argument(
        "--is_patch_size",
        type=int,
        default=is_conf.get("patch_size", DEFAULT_IN_PATCH_SIZE),
    )
    parser.add_argument(
        "--is_batch_size",
        type=int,
        default=is_conf.get("batch_size", DEFAULT_IN_BATCH_SIZE),
    )
    parser.add_argument(
        "--is_max_area",
        type=float,
        default=is_conf.get("max_area", DEFAULT_IS_MAX_AREA),
    )
    parser.add_argument(
        "--exclude_instanseg",
        action="store_true",
        default=is_conf.get("exclude_instanseg", DEFAULT_EXCLUDE_INSTANSEG),
    )

    # --- Adaptive Threshold Parameters ---
    parser.add_argument(
        "--exclude_adaptive_threshold",
        action="store_true",
        default=adaptive_conf.get(
            "exclude_adaptive_threshold", DEFAULT_EXCLUDE_ADAPTIVE
        ),
    )

    # ---------------------------------------------------------
    # 4. PARSE & INJECT COMPLEX OBJECTS
    # ---------------------------------------------------------
    args = parser.parse_args()

    # Handle the complex list of dicts manually (CLI cannot easily handle list-of-dicts)
    # Strategy: Check YAML first, then fallback to constant.
    if "parameters" in adaptive_conf:
        args.adaptive_threshold_parameters = adaptive_conf["parameters"]
    else:
        args.adaptive_threshold_parameters = DEFAULT_ADAPTIVE_PARAMS

    # Include min_area and max_mean_color in each adaptive threshold param set
    for param_set in args.adaptive_threshold_parameters:
        if "final_min_area" not in param_set:
            param_set["final_min_area"] = args.min_area_threshold
        if "max_mean_color" not in param_set:
            param_set["max_mean_color"] = args.max_mean_color

    # Print summary for verification
    print(f"\n{10 * '='} CONFIG LOADED {10 * '='}")
    pprint.pprint(vars(args))
    print(f"{35 * '='}\n")

    return args
