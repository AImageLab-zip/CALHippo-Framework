from typing import List, Optional

import torch
import torch.nn as nn


class NormalizedClassFFTLoss(nn.Module):
    def __init__(self, class_weights: Optional[List[float]] = None):
        """
        Calculates the 2D Fast Fourier Transform (FFT) loss to prevent spectral bias (smearing).
        Adapted for high-res (128x128) outputs by using orthonormal transforms and spatial means.
        """
        super().__init__()
        # Removed eps since we are dropping the denominator division
        self.class_weights = class_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = preds.shape

        # 1. Mathematically Correct 2D FFT
        # THE FIX: norm="ortho" is strictly required here to prevent the energy
        # from inflating by H*W (16,384) in the frequency domain.
        preds_fft = torch.fft.fft2(preds, dim=(-2, -1), norm="ortho")
        targets_fft = torch.fft.fft2(targets, dim=(-2, -1), norm="ortho")

        # 2. Calculate Squared Error in the Frequency Domain PER CHANNEL
        # [cite_start]We compute the MSE of the real and imaginary components[cite: 403, 405].
        real_err = (preds_fft.real - targets_fft.real) ** 2
        imag_err = (preds_fft.imag - targets_fft.imag) ** 2

        # 3. THE FIX: Spatial Mean, NO gt_sums division
        # We use .mean(dim=(2, 3)) instead of .sum() to safely average the error
        # across the 16,384 pixels, making the loss scale-invariant.
        fft_err = (real_err + imag_err).mean(dim=(2, 3))  # Shape: (B, C)

        # 4. Apply Explicit Class Weights
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights, device=preds.device, dtype=preds.dtype
            )
            fft_err = fft_err * weights.view(1, C)

        # Average equally across the batch and the 3 channels
        return fft_err.mean()


class NormalizedClassPhaseInvariantFFTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_size=16,
        stride=16,
        loss_type="l1",
        eps=1e-8,
        eps_norm=1.0,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Patch-wise FFT loss adapted for class-normalized dense regression.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.loss_type = loss_type
        self.eps = eps

        # Added to integrate with your normalization pipeline
        self.eps_norm = eps_norm
        self.class_weights = class_weights

    def _extract_patches(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # shape: (B, C, nH, nW, patch_H, patch_W)
        return x.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Unfold into patches
        input_patches = self._extract_patches(input)
        target_patches = self._extract_patches(target)

        B, C, nH, nW, ph, pw = input_patches.shape
        N = nH * nW  # Number of patches

        # Reshape patches to (B, C, N, ph, pw)
        input_patches = input_patches.contiguous().view(B, C, N, ph, pw)
        target_patches = target_patches.contiguous().view(B, C, N, ph, pw)

        # FFT2 over each patch
        input_fft = torch.fft.fft2(input_patches, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.fft2(target_patches, dim=(-2, -1), norm="ortho")

        # Magnitude
        input_mag = torch.sqrt(input_fft.real**2 + input_fft.imag**2 + self.eps)
        target_mag = torch.sqrt(target_fft.real**2 + target_fft.imag**2 + self.eps)

        # Compute spectral distance
        if self.loss_type == "l1":
            loss = torch.abs(input_mag - target_mag)
        elif self.loss_type == "l2":
            loss = (input_mag - target_mag) ** 2

        # ====================================================================
        # MINIMAL EDITS START HERE
        # ====================================================================

        # 1. Sum over patches and spatial frequencies to get total error PER CHANNEL
        # Original: loss = loss.view(B, C, N, -1).mean(dim=-1).mean(dim=(1, 2))
        fft_err = loss.sum(dim=(2, 3, 4))  # Shape becomes (B, C)

        # 2. Safe Normalization by Ground Truth count
        gt_sums = target.sum(dim=(2, 3))
        gt_sums = torch.clamp(gt_sums, min=10.0)  # Prevent explosion

        normalized_loss = fft_err / (gt_sums + self.eps_norm)

        # 3. Apply Explicit Class Weights
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights, device=input.device, dtype=input.dtype
            )
            normalized_loss = normalized_loss * weights.view(1, C)

        # Average equally across batch and channels
        return normalized_loss.mean()
