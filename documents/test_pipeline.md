# Test Pipeline

This is the maintained smoke test for checking the public pipeline through LR
density dataset creation without running GT evaluation or point-cloud exports.

It uses two image IDs, `3305` and `3348`, so `GroupShuffleSplit` validates the
real train/test split path. It runs `RCA3` only, with all segmentation models
enabled, then runs released UNI2-h classifier inference and density dataset
creation.

Run it from the repository root:

```bash
uv run python scripts/test_pipeline.py
```

By default the script uses `data_temp` and deletes that folder after a successful
run. Keep outputs for inspection with:

```bash
uv run python scripts/test_pipeline.py --keep-data-root
```

If a previous kept run already has all expected outputs, re-check it quickly:

```bash
uv run python scripts/test_pipeline.py --skip-downloads --reuse-existing --keep-data-root
```

## What It Tests

- Data setup downloads for surfaces, model weights, and HR/LR IDs `3305 3348`.
- HR preprocessing into all-region crops and RCA3 single-region crops.
- LR preprocessing for the two non-contiguous IDs.
- RCA3 all-model segmentation using Cellpose, StarDist, HoverNet, InstanSeg, and adaptive threshold variants.
- Classification inference with `ml_classifier_logistic_encoder_uni2h`.
- Density dataset creation with one image group in `train` and one in `test`.

## What It Does Not Test

- Classification training from GT annotations.
- All-region or full-ID-range segmentation/classification.
- Density model training.
- `gt_predict_eval.py`, because GT density array preparation is not implemented yet.
- Optional point-cloud rotating GIF and STL/volume export.

## Notes

- The released classifier requires gated access to `MahmoodLab/UNI2-h`; authenticate with `hf auth login` or set `HF_TOKEN` before running if the model is not already cached.
- Segmentation and classification use GPU when available and can take several minutes. CUDA OOM is possible on small GPUs.
- The smoke test lowers segmentation batch sizes and uses a classification batch size of `8` to reduce OOM risk.
- RCA3-only smoke density creation uses relaxed ROI patch thresholds (`--min-intersection 0.0` and `--min-roi-patch-area-ratio 0.0`) because the two RCA3 test ROIs are small enough that default full-dataset thresholds can produce zero patches.
