# HoverNet Inference Lite

A lightweight, standalone version of HoverNet for nuclei segmentation inference on image tiles.

## Features

- **Minimal dependencies**: Only essential packages required for inference
- **Self-contained**: No dependency on the full HoverNet repository
- **Tile mode only**: Optimized for tile-based inference (WSI mode removed)
- **Easy integration**: Copy this folder into your project

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
export MPLBACKEND=Agg
python run_infer.py \
    --gpu=0 \
    --model_path=/path/to/hovernet_checkpoint.tar \
    --model_mode=original \
    --nr_types=0 \
    tile \
    --input_dir=/path/to/input/images/ \
    --output_dir=/path/to/output/
```

### Arguments

#### Global Options:
- `--gpu`: GPU ID(s) to use (default: 0)
- `--model_path`: Path to the HoverNet checkpoint (.tar file)
- `--model_mode`: Model architecture - `original` or `fast` (default: fast)
- `--nr_types`: Number of nuclei types (0 = no type classification)
- `--type_info_path`: Path to JSON file with type-color mapping
- `--nr_inference_workers`: Number of inference workers (default: 8)
- `--nr_post_proc_workers`: Number of post-processing workers (default: 16)
- `--batch_size`: Batch size per GPU (default: 32)

#### Tile Mode Options:
- `--input_dir`: Input directory containing image files
- `--output_dir`: Output directory for results
- `--mem_usage`: Memory usage fraction for caching (default: 0.2)
- `--draw_dot`: Draw centroid dots on overlay
- `--save_qupath`: Save QuPath-compatible output (.tsv)
- `--save_raw_map`: Save raw prediction maps

## Output

The tool generates three output folders:
- `json/`: Instance segmentation results in JSON format
- `mat/`: MATLAB-compatible .mat files with instance maps
- `overlay/`: Visualization overlays with segmented nuclei

If `--save_qupath` is enabled:
- `qupath/`: QuPath v0.2.3 compatible .tsv files

## Directory Structure

```
hovernet_infer_lite/
├── run_infer.py           # Main entry point
├── requirements.txt       # Minimal dependencies
├── convert_format.py      # QuPath export utilities
├── misc/
│   ├── utils.py           # Utility functions
│   └── viz_utils.py       # Visualization utilities
├── dataloader/
│   └── infer_loader.py    # Data loading utilities
├── models/
│   └── hovernet/
│       ├── net_desc.py    # HoverNet architecture
│       ├── net_utils.py   # Neural network building blocks
│       ├── utils.py       # Model utilities
│       ├── run_desc.py    # Inference step
│       └── post_proc.py   # Post-processing
└── infer/
    ├── base.py            # Base inference manager
    └── tile.py            # Tile-based inference
```

## Dependencies

Core packages (from requirements.txt):
- torch >= 1.8.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- scipy >= 1.6.0
- scikit-image >= 0.18.0
- psutil >= 5.8.0
- tqdm >= 4.60.0
- docopt >= 0.6.2
- matplotlib >= 3.3.0

## Notes

- This is a read-only inference tool; training functionality is not included
- The `original` mode requires 270x270 input patches with 80x80 outputs
- The `fast` mode requires 256x256 input patches with 164x164 outputs
- Input images should be RGB format (PNG, JPEG, etc.)

## License

Same as the original HoverNet repository.
