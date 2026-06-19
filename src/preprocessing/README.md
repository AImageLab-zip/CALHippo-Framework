# Pre-Processing

This submodule contains the preprocessing utilities used to turn the raw BigBrain inputs into the cropped images and ROI annotations consumed by the rest of the repository.

For the maintained data download and folder-creation workflow, use
`scripts/setup_data.py` and read:

- `README.md`
- `documents/data_setup.md`

For the maintained reproducibility and inference path after data setup, read:

- `documents/pipeline.md`

## Data Needed

The maintained data-root convention stores raw preprocessing inputs under
`<DATA_ROOT>/raw/`:

1. Low-resolution (LR) coronal images in `.mnc` format under
   `<DATA_ROOT>/raw/low_res/`
   https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/2D_Final_Sections/Coronal/Minc

2. High-resolution (HR) aligned images in `.tif` format under
   `<DATA_ROOT>/raw/high_res/`, with one affine `.json` file per image
   https://data-proxy.ebrains.eu/api/v1/buckets/p22717-hbp-d000070_BigBrain-selected_1um_scans_pub/v1.0/aligned/
   The maintained setup path downloads per-ID `.tif` and `_affine.json` pairs.

3. Hippocampal segmentation surfaces in `.surf.gii` format under
   `<DATA_ROOT>/raw/masks/3dVolumes_SegmentationMasks_40um/`
   https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/Hippocampus_Segmentation/gii/

Maintained region names are `RCA1`, `RCA2`, `RCA3`, and `RCA4`.

## Main Files in `preprocessing`

- `extract_crops_and_coords_HR.py`
  Main HR preprocessing entry point. For each full HR image, it slices the
  hippocampal 3D surfaces at the image world y position, maps the resulting
  contours to HR pixels, computes one crop covering the selected regions, and
  saves all-region outputs under `<DATA_ROOT>/input/all_regions/high_res/`:
  - `<image_id>_HR_crop.tif`
  - `<image_id>_bbox_hr.json`
  - `<image_id>_contours_hr.geojson`

- `extract_crops_and_coords_LR.py`
  Main LR preprocessing entry point. It performs the same surface-slicing logic
  for LR coronal slices, using the LR affine convention and the LR-specific
  image_id -> y_world rule. It saves all-region outputs under
  `<DATA_ROOT>/input/all_regions/low_res/`:
  - `<image_id>_LR_crop.png`
  - `<image_id>_bbox_lr.json`
  - `<image_id>_contours_lr.geojson`

  Important: the LR bbox is saved in the original full-image LR coordinates,
  while the exported crop and GeoJSON are flipped vertically to match the
  viewing convention used for HR images.

- `extract_hr_region_crops.py`
  Takes the multi-region HR outputs produced by extract_crops_and_coords_HR.py
  and splits them into one crop per region. This is the dataset used later by
  the HR cell segmentation and classification steps. Canonical outputs live
  under `<DATA_ROOT>/input/single_regions/high_res/<REGION>/`.

- `surfaces_utils.py`
  Shared helpers to load .surf.gii surfaces, convert them to PyVista PolyData,
  and slice them with a plane at a given world y coordinate.

- `generate_masks_utils.py`
  Shared logic used by both HR and LR extraction scripts. It maps world-space
  contours into image coordinates, computes the merged crop bbox, groups holes,
  and exports the GeoJSON annotations.

## Typical Workflow

1. Run `extract_crops_and_coords_HR.py` on `<DATA_ROOT>/raw/high_res/`. This creates one large HR crop per image containing the selected hippocampal regions and their contours in `<DATA_ROOT>/input/all_regions/high_res/`.

2. Optionally adjust or review those all-region HR annotations if needed.

3. Run `extract_hr_region_crops.py` to split the HR annotations into one folder per region under `<DATA_ROOT>/input/single_regions/high_res/<REGION>/`, with one crop, one bbox, and one GeoJSON per image.

4. Run `extract_crops_and_coords_LR.py` on `<DATA_ROOT>/raw/low_res/`. These LR crops and contours are written to `<DATA_ROOT>/input/all_regions/low_res/` and used later by the density-estimation, LR inference, and point-cloud stages.
