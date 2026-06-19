"""Dataset and transforms for density estimation on tiled WSI patches."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch.utils.data import Dataset


class SimpleCADataset(Dataset):
    """
    Loads paired image patches (.png) and density maps (.npy) for training.

    Expected directory layout under ``root_dir / split``:
    ::

        <split>/
        ├── images/   # *.png  (RGB, 16-bit or 8-bit)
        └── roi_mask/ # *.npy  (uint8, shape HxWxC or CxHxW)
        └── densities/ # *.npy  (float32, shape HxWxC or CxHxW)

    Args:
        root_dir: Path to the root dataset folder (containing ``train`` / ``test``).
        split: Subfolder name — typically ``"train"`` or ``"test"``.
        transform: An ``albumentations.Compose`` pipeline applied to every sample.
        return_filenames: If ``True``, include the stem filename in the output dict.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        return_filenames: bool = False,
        load_roi_masks: bool = False,
        max_pix_value: float = 65535.0,
        channel_to_predict: int | None = None,
    ):
        self.root = Path(root_dir) / split
        self.img_dir = self.root / "images"
        self.dens_dir = self.root / "densities"
        self.roi_dir = self.root / "roi_masks"

        self.filenames = sorted([f.stem for f in self.img_dir.glob("*.png")])
        self.transform = transform
        self.return_filenames = return_filenames
        self.load_roi_masks = load_roi_masks
        self.max_pix_value = max_pix_value  # For normalizing 16-bit images to [0, 1]
        self.channel_to_predict = channel_to_predict

        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        logger.debug(
            f"SimpleCADataset [{split}]: {len(self.filenames)} samples from {self.root}"
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]

        img_path = self.img_dir / f"{fname}.png"
        roi_path = self.roi_dir / f"{fname}_roi_mask.npy"
        dens_path = self.dens_dir / f"{fname}.npy"

        # Load image — IMREAD_UNCHANGED preserves 16-bit depth
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (
            image.astype(np.float32) / self.max_pix_value
        )  # normalize to [0, 1] range for 16-bit images

        # Load density mask (raw float counts)
        mask = np.load(str(dens_path)).astype(np.float32)

        # Select single channel if specified (for single-class prediction)
        if self.channel_to_predict is not None:
            if mask.ndim != 3:
                raise ValueError(
                    f"Expected density map with shape (H, W, C) when selecting a channel, got {mask.shape} for {dens_path}."
                )
            if not 0 <= self.channel_to_predict < mask.shape[2]:
                raise IndexError(
                    f"channel_to_predict={self.channel_to_predict} is out of range for density map {dens_path} with {mask.shape[2]} channels."
                )
            mask = mask[:, :, self.channel_to_predict : self.channel_to_predict + 1]

        # Load ROI mask if needed
        roi_mask = None
        if self.load_roi_masks:
            roi_mask = np.load(str(roi_path)).astype(np.uint8)
            # Ensure ROI mask has channel dimension for albumentations (H, W) -> (H, W, 1)
            if roi_mask.ndim == 2:
                roi_mask = roi_mask[:, :, np.newaxis]

        # Apply augmentations
        if self.transform:
            aug_dict = {"image": image, "mask": mask}
            if roi_mask is not None:
                aug_dict["roi_mask"] = roi_mask

            augmented = self.transform(**aug_dict)
            image = augmented["image"]
            mask = augmented["mask"]
            if roi_mask is not None:
                roi_mask = augmented["roi_mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            if roi_mask is not None:
                # Remove channel dimension if added: (H, W, 1) -> (H, W)
                if roi_mask.ndim == 3 and roi_mask.shape[-1] == 1:
                    roi_mask = roi_mask.squeeze(-1)
                roi_mask = torch.from_numpy(roi_mask).long()

        # Ensure mask is (C, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask = mask.permute(2, 0, 1)

        # Ensure ROI mask is proper shape after augmentation
        if roi_mask is not None:
            # After augmentation, roi_mask could be (H, W) or (H, W, 1) tensor
            if roi_mask.ndim == 3 and roi_mask.shape[-1] == 1:
                # Remove singleton channel: (H, W, 1) -> (H, W)
                roi_mask = roi_mask.squeeze(-1)
            if roi_mask.ndim == 2:
                # Add batch-like channel dimension: (H, W) -> (1, H, W)
                roi_mask = roi_mask.unsqueeze(0)
            # Clone to create a clean tensor with its own storage for DataLoader batching
            roi_mask = roi_mask.clone()

        sample: Dict[str, torch.Tensor] = {
            "filename": fname,
            "image": image,
            "mask": mask,
        }

        if self.load_roi_masks:
            sample["roi_mask"] = roi_mask

        if self.return_filenames:
            sample["filename"] = fname

        return sample


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------
# Processed on the all ca dataset (train set) with 16-bit images normalized to [0, 1] by dividing by 65535.0.
# Final Mean: tensor([0.7640, 0.7640, 0.7640], dtype=torch.float64)
# Final Std: tensor([0.0799, 0.0799, 0.0799], dtype=torch.float64)


def get_transforms(
    img_size: int = 128,
    norm_mean: Tuple[float, ...] = (0.7640, 0.7640, 0.7640),
    norm_std: Tuple[float, ...] = (0.0799, 0.0799, 0.0799),
    fill_value: float = 1.0,
    aug_level: str = "basic",
    load_roi_masks: bool = False,
) -> Tuple[A.Compose, A.Compose]:
    """
    Build train / val augmentation pipelines.

    Args:
        img_size: Minimum spatial dimension (pad if smaller).
        norm_mean: Per-channel mean for ``A.Normalize``.
        norm_std: Per-channel std for ``A.Normalize``.
        fill_value: Padding constant for images.
        aug_level: "basic", "medium", or "full".
        load_roi_masks: If True, configure transforms to handle roi_mask.
    """
    if aug_level not in ["basic", "medium", "full"]:
        raise ValueError(
            f"aug_level must be 'basic', 'medium', or 'full', got {aug_level}"
        )

    # Configure additional targets if loading ROI masks
    additional_targets = {}
    if load_roi_masks:
        additional_targets["roi_mask"] = "mask"

    train_augs = []

    # ---------------------------------------------------------
    # 1. Medium: Safe Geometric (Preserves physical pixel area)
    # ---------------------------------------------------------
    if aug_level in ["medium", "full"]:
        train_augs.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # ---------------------------------------------------------
    # 2. Full: Textural & Coarse Dropout
    # ---------------------------------------------------------
    if aug_level == "full":
        train_augs.extend(
            [
                # Safe 16-bit alternative to ColorJitter
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.7
                ),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.5), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=16,
                    max_width=16,
                    fill_value=fill_value,
                    mask_fill_value=0,  # CRITICAL: Zeroes out the density target securely
                    p=0.3,
                ),
            ]
        )

    # ---------------------------------------------------------
    # 3. Base Processing (Always applied last)
    # ---------------------------------------------------------
    base_processing = [
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=fill_value,  # 'value' instead of 'fill' in newer albumentations
            fill_mask=0,  # 'mask_value' instead of 'fill_mask'
        ),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=fill_value),
        ToTensorV2(transpose_mask=True),  # Ensure mask is (C, H, W)
    ]

    train_transform = A.Compose(
        train_augs + base_processing,
        additional_targets=additional_targets,
    )

    # Validation is strictly deterministic
    val_transform = A.Compose(
        base_processing,
        additional_targets=additional_targets,
    )

    return train_transform, val_transform


_WSI_ID_RE = re.compile(r"(\d{4,})")


def get_groups(dataset: SimpleCADataset) -> np.ndarray:
    """
    Extract WSI identifiers from filenames for ``GroupKFold``.

    Supports multiple naming conventions:

    * ``RCA1_patch_25_100``           → ``RCA1``   (legacy, no 4-digit number)
    * ``2845_HR_crop_x_0_y_0``       → ``2845``
    * ``RCA1_2845_x_128_y_0``        → ``2845``
    * ``3096_HR_crop_roi_0_x_0_y_16``→ ``3096``

    The heuristic is: extract the first 4+-digit number in the filename (the
    brain-section ID).  If no such number exists, fall back to the first
    underscore-delimited token.
    """
    groups: list[str] = []
    for f in dataset.filenames:
        m = _WSI_ID_RE.search(f)
        groups.append(m.group(1) if m else f.split("_")[0])
    return np.array(groups)
