from __future__ import annotations

import numpy as np
import torch
from loguru import logger


def create_gaussian_mask(
    patch_size: int = 128,
    sigma: float = 24.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generates a 2D Gaussian mask for window blending.
    Center is 1.0, decaying to near 0.0 at the edges.
    """
    coords = torch.arange(patch_size, dtype=torch.float32, device=device)

    center = (patch_size - 1) / 2.0
    coords -= center

    y, x = torch.meshgrid(coords, coords, indexing="ij")

    mask = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return mask


def predict_density_map(
    wsi_tensor: torch.Tensor,
    model: torch.nn.Module,
    patch_size: int,
    stride: int,
    num_classes: int,
    device: str,
    inference_batch_size: int = 32,
) -> torch.Tensor:
    """
    Predict a density map for the entire WSI by sliding a window across it,
    applying the model to each patch, and stitching the results together with
    a Gaussian weighting to smooth the overlaps.
    """
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be >= 1")

    _, _, height, width = wsi_tensor.shape

    global_density = torch.zeros(
        (1, num_classes, height, width), dtype=torch.float32, device=device
    )
    global_weight = torch.zeros(
        (1, num_classes, height, width), dtype=torch.float32, device=device
    )

    sigma = (patch_size // 2) // 3
    gaussian_mask = create_gaussian_mask(patch_size, sigma=sigma, device=device)
    gaussian_mask = gaussian_mask.view(1, 1, patch_size, patch_size)

    max_h = height - patch_size
    max_w = width - patch_size

    # Extract all the possible patches and their anchors
    h_anchors = list(range(0, max_h, stride)) + [max_h]
    w_anchors = list(range(0, max_w, stride)) + [max_w]
    patch_anchors = [(y, x) for y in h_anchors for x in w_anchors]
    patches = [
        wsi_tensor[:, :, y : y + patch_size, x : x + patch_size]
        for y, x in patch_anchors
    ]

    model.eval()
    with torch.no_grad():
        for batch_start_idx in range(0, len(patches), inference_batch_size):
            # Extract batch data and move to device
            batch_end_idx = batch_start_idx + inference_batch_size
            patch_batch = torch.cat(patches[batch_start_idx:batch_end_idx], dim=0).to(device)
            
            # Predict batch
            pred_density_batch = model(patch_batch)

            # Save results
            for pred_density, (y, x) in zip(
                pred_density_batch,
                patch_anchors[batch_start_idx:batch_end_idx],
                strict=True,
            ):
                pred_density = pred_density.unsqueeze(dim=0)
                global_density[:, :, y : y + patch_size, x : x + patch_size] += (
                    pred_density * gaussian_mask
                )
                global_weight[:, :, y : y + patch_size, x : x + patch_size] += (
                    gaussian_mask
                )

        logger.debug(f"Total patches processed: {len(patches)}")
        final_stitched_map = global_density / (global_weight + 1e-8)

    return final_stitched_map


def unpad_density_map(
    final_stitched_map: torch.Tensor,
    pad: dict[str, int],
) -> torch.Tensor:
    top, bottom, left, right = (
        pad["top"],
        pad["bottom"],
        pad["left"],
        pad["right"],
    )

    h_padded, w_padded = final_stitched_map.shape[2], final_stitched_map.shape[3]

    y_end = h_padded - bottom
    x_end = w_padded - right

    return final_stitched_map[:, :, top:y_end, left:x_end]


def sample_discrete_density_numpy(density_mask: np.ndarray) -> np.ndarray:
    """
    Converts a continuous (H, W, C) density map into a discrete map of integer counts
    using NumPy's native multinomial sampler and probabilistic rounding.

    Args:
        density_mask: NumPy array of shape (H, W, C) containing
            continuous density values.

    Returns:
        discrete_map: NumPy array of shape (H, W, C) containing discrete integer counts,
        where cells can stack in the same pixel.
    """
    height, width, channels = density_mask.shape

    discrete_map = np.zeros((height, width, channels), dtype=np.int32)

    clean_density = np.clip(density_mask, a_min=0.0, a_max=None)

    for channel_idx in range(channels):
        channel_map = clean_density[:, :, channel_idx]
        total_mass = channel_map.sum()

        if total_mass <= 1e-6:
            logger.debug(
                "Channel "
                f"{channel_idx} is empty (total mass: {total_mass:.6f}). "
                "Skipping."
            )
            continue

        int_mass = int(total_mass)
        frac_mass = total_mass - int_mass

        extra_cell = 1 if np.random.rand() < frac_mass else 0
        n_samples = int_mass + extra_cell

        if n_samples == 0:
            continue

        flat_map = channel_map.flatten()
        pmf = flat_map / total_mass

        pmf[-1] = 1.0 - pmf[:-1].sum()
        pmf = np.clip(pmf, a_min=0.0, a_max=None)

        flat_counts = np.random.multinomial(n_samples, pmf)

        discrete_map[:, :, channel_idx] = flat_counts.reshape((height, width))

    return discrete_map
