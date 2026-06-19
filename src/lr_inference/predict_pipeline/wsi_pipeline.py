from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from src.lr_inference.predict_pipeline.data_loading import (
    extract_roi_mask_from_geojson,
    load_geojson_data,
    load_low_res_wsi,
)
from src.lr_inference.predict_pipeline.prediction import (
    predict_density_map,
    sample_discrete_density_numpy,
    unpad_density_map,
)
from src.lr_inference.predict_pipeline.visualization import (
    save_prediction_visualization,
)


def run_prediction_for_wsi(
    wsi_entry: dict[str, Path | None],
    model: torch.nn.Module,
    val_transform: Any,
    max_pix_value: float,
    patch_size: int,
    stride: int,
    num_classes: int,
    device: str,
    roi_class: str,
    ca_list: list[str],
    effective_class_list: list[str],
    save_visualizations: bool,
    inference_batch_size: int = 32,
) -> dict[str, Any]:
    """Run the full prediction pipeline for a single LR WSI."""
    wsi_path = wsi_entry["wsi_path"]
    if wsi_path is None:
        raise ValueError("wsi_entry['wsi_path'] must not be None")

    logger.info(f"Processing WSI: {wsi_path}")

    wsi_id = wsi_path.stem.split(sep="_")[0]
    roi_mask = None
    roi_density_array = None

    # Load the WSI tensor
    wsi_tensor, pad = load_low_res_wsi(
        wsi_path=wsi_path,
        transforms=val_transform,
        max_pix_value=max_pix_value,
    )
    wsi_tensor = wsi_tensor.unsqueeze(dim=0)

    logger.debug(f"Padding info: {pad}")
    logger.debug(f"WSI tensor shape after padding: {wsi_tensor.shape}")

    # Predict the density map
    wsi_density_map = predict_density_map(
        wsi_tensor=wsi_tensor.to(device),
        model=model,
        patch_size=patch_size,
        num_classes=num_classes,
        stride=stride,
        device=device,
        inference_batch_size=inference_batch_size,
    )

    # Save full density map
    wsi_density_map = unpad_density_map(wsi_density_map, pad)
    full_density_array = wsi_density_map.squeeze_(dim=0).permute(1, 2, 0).cpu().numpy()

    full_density_preds_path = wsi_entry["full_density_preds_path"]
    if full_density_preds_path is not None:
        np.save(full_density_preds_path, full_density_array)

    geojson_path = wsi_entry["geojson_path"]
    if geojson_path:
        geojson_data = load_geojson_data(geo_path=geojson_path)

        # Extract ROI mask, apply to density map, and save it
        if wsi_entry["roi_density_preds_path"]:
            roi_mask = extract_roi_mask_from_geojson(
                geojson=geojson_data,
                image_shape=full_density_array.shape[:2],
                roi_class=roi_class,
            )

            np.save(wsi_entry["roi_mask_output_path"], roi_mask)

            roi_mask = roi_mask[..., np.newaxis]
            roi_density_array = full_density_array * roi_mask

            np.save(wsi_entry["roi_density_preds_path"], roi_density_array)

        # Extract CA areas bound and save it
        if wsi_entry["ca_areas_output_path"]:
            ca_area_map = np.zeros_like(full_density_array[..., 0], dtype=np.uint8)

            for ca_area in ca_list:
                ca_id = int(ca_area[-1])

                current_ca_mask = extract_roi_mask_from_geojson(
                    geojson=geojson_data,
                    image_shape=full_density_array.shape[:2],
                    roi_class=ca_area,
                )

                current_ca_mask = current_ca_mask.astype(int) * ca_id

                ca_map_free_spaces = ca_area_map == 0
                current_ca_mask = current_ca_mask * ca_map_free_spaces
                ca_area_map += current_ca_mask.astype(np.uint8)

            np.save(wsi_entry["ca_areas_output_path"], ca_area_map)

    pred_array = (
        roi_density_array if roi_density_array is not None else full_density_array
    )

    # Sample and save discrete points map
    discrete_points_map = sample_discrete_density_numpy(pred_array)
    points_preds_path = wsi_entry["points_preds_path"]
    if points_preds_path is not None:
        np.save(points_preds_path, discrete_points_map)
        logger.info(f"Saved discrete points map to {points_preds_path}")

    # Save visualizations if enabled
    if save_visualizations:
        orig_img_np, _ = load_low_res_wsi(
            wsi_path=wsi_path,
            max_pix_value=max_pix_value,
        )

        if orig_img_np.shape[:2] != full_density_array.shape[:2]:
            logger.warning(
                f"Base image shape {orig_img_np.shape} differs from density shape "
                f"{full_density_array.shape}. Resizing image to match density."
            )
            orig_img_np = cv2.resize(
                orig_img_np,
                (full_density_array.shape[1], full_density_array.shape[0]),
            )

        save_prediction_visualization(
            wsi_id=wsi_id,
            original_image=orig_img_np,
            full_density=full_density_array,
            sampled_points=discrete_points_map,
            class_list=effective_class_list,
            output_dir=Path(full_density_preds_path).parent,
            roi_mask=roi_mask,
            masked_density=roi_density_array,
        )

    return {
        "wsi_id": wsi_id,
        "full_density_array": full_density_array,
        "pred_array": pred_array,
        "roi_mask": roi_mask,
        "roi_density_array": roi_density_array,
        "discrete_points_map": discrete_points_map,
    }
