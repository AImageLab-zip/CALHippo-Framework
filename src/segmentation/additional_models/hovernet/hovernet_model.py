"""
HoverNet Model Wrapper for integration with the multimodel inference pipeline.

This module provides a simple class-based API to load HoverNet models and run
inference on images (numpy arrays), returning instance segmentation masks and
instance information compatible with the segmentation inference pipeline.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models.hovernet import net_desc, post_proc


def convert_pytorch_checkpoint(net_state_dict: dict) -> dict:
    """Convert data-parallel checkpoint to single GPU mode."""
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        print("WARNING: Detect checkpoint saved in data-parallel mode. Converting...")
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


class HoverNetModel:
    """
    HoverNet model wrapper for inference on single images.

    This class provides a simple API compatible with the multimodel inference pipeline.
    It loads the HoverNet model once and provides an `eval` method that accepts an
    image as a numpy array and returns the instance segmentation mask and metadata.

    Attributes:
        model_path: Path to the HoverNet checkpoint (.tar file)
        model_mode: Either 'original' or 'fast' (default: 'original')
        nr_types: Number of nuclei types to predict (default: None for binary segmentation)
        batch_size: Batch size for inference (default: 32)

    Example:
        >>> model = HoverNetModel(
        ...     model_path="/path/to/checkpoint.tar",
        ...     model_mode="original",
        ...     nr_types=None,
        ... )
        >>> image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        >>> masks, metadata = model.eval(image)
        >>> # masks is an np.ndarray with instance labels
        >>> # metadata is a dict with 'inst_info' containing per-instance information
    """

    def __init__(
        self,
        model_path: str,
        model_mode: str = "original",
        nr_types: Optional[int] = None,
        batch_size: int = 32,
        gpu: bool = True,
    ):
        """
        Initialize HoverNet model.

        Args:
            model_path: Path to the HoverNet checkpoint file (.tar format)
            model_mode: Model architecture mode - 'original' (270x270 patches) or
                       'fast' (256x256 patches). Default: 'original'
            nr_types: Number of nuclei types for classification.
                     None for binary segmentation (no type prediction).
            batch_size: Batch size for inference. Default: 32
            gpu: Whether to use GPU. Default: True
        """
        self.model_path = model_path
        self.model_mode = model_mode
        self.nr_types = nr_types
        self.batch_size = batch_size
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

        # Patch sizes based on model mode
        if model_mode == "fast":
            self.patch_input_shape = 256
            self.patch_output_shape = 164
        else:  # original
            self.patch_input_shape = 270
            self.patch_output_shape = 80

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the HoverNet model and checkpoint."""
        # Create model
        model_creator = net_desc.create_model
        self.net = model_creator(nr_types=self.nr_types, mode=self.model_mode)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        saved_state_dict = checkpoint["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        self.net.load_state_dict(saved_state_dict, strict=True)
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.to(self.device)
        self.net.eval()

        # Store post-processing function reference
        self.post_proc_func = post_proc.process

    def _prepare_patching(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare patch information for tile processing.

        Args:
            img: Input image (H, W, C)

        Returns:
            Tuple of (padded_image, patch_info, top_corner_offset)
        """
        win_size = self.patch_input_shape
        msk_size = step_size = self.patch_output_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = img.shape[0]
        im_w = img.shape[1]

        last_h, _ = get_last_steps(im_h, msk_size, step_size)
        last_w, _ = get_last_steps(im_w, msk_size, step_size)

        diff = win_size - step_size
        padt = padl = diff // 2
        padb = last_h + win_size - im_h
        padr = last_w + win_size - im_w

        img = np.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

        coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
        coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
        row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
        col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
        coord_y, coord_x = np.meshgrid(coord_y, coord_x)
        row_idx, col_idx = np.meshgrid(row_idx, col_idx)
        coord_y = coord_y.flatten()
        coord_x = coord_x.flatten()
        row_idx = row_idx.flatten()
        col_idx = col_idx.flatten()

        patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
        return img, patch_info, [padt, padl]

    def _infer_step(self, batch_data: torch.Tensor) -> np.ndarray:
        """
        Run inference on a batch of patches.

        Args:
            batch_data: Input batch of image patches (B, H, W, C)

        Returns:
            Concatenated prediction output (np, hv, and optionally tp)
        """
        patch_imgs_gpu = batch_data.to(self.device).type(torch.float32)
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            pred_dict = self.net(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            pred_output = torch.cat(list(pred_dict.values()), -1)

        return pred_output.cpu().numpy()

    def _assemble_predictions(
        self,
        patch_outputs: List[Tuple[np.ndarray, np.ndarray]],
        src_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Assemble patch predictions into full image prediction.

        Args:
            patch_outputs: List of (patch_info, patch_data) tuples
            src_shape: Original image shape (H, W)

        Returns:
            Full prediction map
        """
        # Sort by position
        patch_outputs = sorted(patch_outputs, key=lambda x: [x[0][0], x[0][1]])
        patch_info_list, patch_data_list = zip(*patch_outputs)

        patch_shape = np.squeeze(patch_data_list[0]).shape
        ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
        axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

        nr_row = max([x[2] for x in patch_info_list]) + 1
        nr_col = max([x[3] for x in patch_info_list]) + 1

        pred_map = np.concatenate(patch_data_list, axis=0)
        pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
        pred_map = np.transpose(pred_map, axes)
        pred_map = np.reshape(
            pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
        )
        pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

        return pred_map

    def eval(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format, uint8.
                   The image should be a crop/tile, not a whole slide image.

        Returns:
            Tuple of:
                - masks: Instance segmentation map (H, W) where each unique
                         integer > 0 represents a different cell instance
                - metadata: Dictionary containing:
                    - 'inst_info': Dict mapping instance_id to info dict with:
                        - 'centroid': (x, y) centroid coordinates
                        - 'contour': (N, 2) array of contour points
                        - 'bbox': [[rmin, cmin], [rmax, cmax]]
                        - 'type': nuclei type (if nr_types is set)
                        - 'type_prob': type probability (if nr_types is set)
                    - 'prob': List of probability values for each instance
                              (estimated from the prediction map)

        Example:
            >>> masks, metadata = model.eval(image)
            >>> print(f"Found {masks.max()} instances")
            >>> for inst_id, info in metadata["inst_info"].items():
            ...     print(f"Instance {inst_id}: centroid={info['centroid']}")
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        src_shape = image.shape[:2]

        # Prepare patches
        padded_img, patch_info, top_corner = self._prepare_patching(image)

        # Extract patches and run inference
        accumulated_outputs = []
        num_patches = len(patch_info)

        for batch_start in range(0, num_patches, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_patches)
            batch_patches = []
            batch_info = []

            for idx in range(batch_start, batch_end):
                info = patch_info[idx]
                patch = padded_img[
                    info[0] : info[0] + self.patch_input_shape,
                    info[1] : info[1] + self.patch_input_shape,
                ]
                batch_patches.append(patch)
                batch_info.append(info)

            # Stack and run inference
            batch_tensor = torch.from_numpy(np.stack(batch_patches, axis=0))
            batch_output = self._infer_step(batch_tensor)

            # Store outputs
            for i, info in enumerate(batch_info):
                accumulated_outputs.append((info, batch_output[i : i + 1]))

        # Assemble full prediction map
        pred_map = self._assemble_predictions(accumulated_outputs, src_shape)

        # Run post-processing
        pred_inst, inst_info_dict = self.post_proc_func(
            pred_map, nr_types=self.nr_types, return_centroids=True
        )

        # Extract probability estimates from the prediction map
        # The first channel of pred_map is the nuclei probability
        prob_list = []
        if inst_info_dict:
            np_channel = pred_map[..., 0] if pred_map.ndim == 3 else pred_map
            for inst_id in sorted(inst_info_dict.keys()):
                inst_mask = pred_inst == inst_id
                if np.any(inst_mask):
                    # Use median probability within the instance region
                    inst_prob = float(np.median(np_channel[inst_mask]))
                    prob_list.append(inst_prob)
                else:
                    prob_list.append(0.5)  # Default probability

        metadata = {
            "inst_info": inst_info_dict if inst_info_dict else {},
            "prob": prob_list,
        }

        return pred_inst, metadata

    def __repr__(self) -> str:
        return (
            f"HoverNetModel(model_path='{self.model_path}', "
            f"model_mode='{self.model_mode}', nr_types={self.nr_types}, "
            f"batch_size={self.batch_size}, device='{self.device}')"
        )
