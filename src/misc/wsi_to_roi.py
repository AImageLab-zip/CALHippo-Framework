import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyvips
from natsort import natsorted
from PIL import Image
from shapely.geometry import shape
from tiffslide import TiffSlide
from tqdm import tqdm


def numpy_to_pyvips(img_np):
    """
    Converts Numpy array to Pyvips and forces sRGB to fix color issues.
    """
    height, width, bands = img_np.shape
    linear = img_np.reshape(width * height * bands)

    img_vips = pyvips.Image.new_from_memory(linear.data, width, height, bands, "uchar")
    img_vips = img_vips.copy(interpretation="srgb")
    return img_vips


def extract_roi_images(
    wsi_path: Path, mask_json_path: Path, output_dir: Path, padding: int = 0
):
    # 1. Load GeoJSON
    try:
        with mask_json_path.open("r") as f:
            roi_data = json.load(f)
    except Exception as e:
        print(f"Error loading mask {mask_json_path}: {e}")
        return

    roi_features = [f for f in roi_data.get("features", []) if f.get("geometry")]
    if not roi_features:
        return

    # 2. Open WSI with TiffSlide (Fixes "weird" colors by using Pillow backend)
    try:
        with TiffSlide(wsi_path) as slide:
            # Get Physical Resolution (Microns Per Pixel)
            try:
                mpp_x = float(slide.properties.get("openslide.mpp-x", 0))
                mpp_y = float(slide.properties.get("openslide.mpp-y", 0))
                # Convert to pixels/mm for Pyvips (1mm = 1000um)
                xres = 1000.0 / mpp_x if mpp_x > 0 else 0
                yres = 1000.0 / mpp_y if mpp_y > 0 else 0
            except:
                xres, yres = 0, 0

            # 3. Iterate ROIs
            for i, feature in enumerate(
                tqdm(roi_features, desc=f"ROIs in {wsi_path.stem}", leave=False)
            ):
                try:
                    roi_geom = shape(feature["geometry"])
                    minx, miny, maxx, maxy = [int(c) for c in roi_geom.bounds]

                    x = max(0, minx - padding)
                    y = max(0, miny - padding)
                    w = (maxx - minx) + (2 * padding)
                    h = (maxy - miny) + (2 * padding)

                    # Boundary Check
                    if x >= slide.dimensions[0] or y >= slide.dimensions[1]:
                        continue

                    # 4. Read Region (Returns PIL Image)
                    roi_pil = slide.read_region((x, y), 0, (w, h))

                    # 5. Handle Alpha/Transparency
                    if roi_pil.mode == "RGBA":
                        background = Image.new("RGB", roi_pil.size, (255, 255, 255))
                        background.paste(roi_pil, mask=roi_pil.split()[3])
                        roi_np = np.array(background)
                    else:
                        roi_np = np.array(roi_pil.convert("RGB"))

                    # 6. Convert to Pyvips
                    vips_crop = numpy_to_pyvips(roi_np)

                    # Set physical resolution metadata
                    if xres > 0:
                        vips_crop = vips_crop.copy(xres=xres, yres=yres)

                    # 7. Save (Original Params, but .tif extension)
                    out_filename = f"{wsi_path.stem}_ROI_{i}.tif"
                    out_path = output_dir / out_filename

                    vips_crop.tiffsave(
                        str(out_path),
                        tile=True,  # Keep: Good for analysis software
                        tile_width=256,
                        tile_height=256,
                        pyramid=True,  # Keep: Good for zooming in QuPath
                        compression="lzw",  # Keep: Lossless
                        bigtiff=True,  # Keep: Safe for large ROIs
                    )

                except Exception as e:
                    print(f"Error extracting ROI {i}: {e}")
                    continue

    except Exception as e:
        print(f"Error opening WSI {wsi_path}: {e}")


def main():
    # Default paths (adjusted to typical use case, change as needed)
    DEFAULT_INPUT_DIR = "data/input/all_regions/high_res"
    DEFAULT_INPUT_MASKS_DIR = "data/input/all_regions/high_res"
    DEFAULT_OUTPUT_DIR = "data/input/single_regions/high_res/RCA3"

    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input_masks_dir", type=str, default=DEFAULT_INPUT_MASKS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img_ext", type=str, default=".tif")
    parser.add_argument("--mask_exts", nargs="+", default=[".geojson"])
    parser.add_argument("--padding", type=int, default=0)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_masks_dir = Path(args.input_masks_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    wsi_files = natsorted(list(input_dir.glob(f"*{args.img_ext}")))
    mask_files = natsorted(
        [
            f
            for f in input_masks_dir.glob("*")
            if f.suffix in args.mask_exts or f.suffix == ".geojson"
        ]
    )

    print(f"Found {len(wsi_files)} WSIs.")

    for wsi_path in tqdm(wsi_files, desc="Processing Slides"):
        wsi_id = wsi_path.stem.split("_")[0]
        matching_masks = [m for m in mask_files if m.stem.startswith(wsi_id)]
        if matching_masks:
            extract_roi_images(wsi_path, matching_masks[0], output_dir, args.padding)


if __name__ == "__main__":
    main()
