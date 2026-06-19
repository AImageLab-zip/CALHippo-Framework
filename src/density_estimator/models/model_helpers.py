import torch.nn.functional as F
from torch import Tensor, nn


def reset_model_weights(layer: nn.Module) -> None:
    """
    Reset learnable parameters of a single layer (if supported).

    Intended for use with ``model.apply(reset_model_weights)``.
    """
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()


def resize_density_tensor(
    density_map: Tensor,
    target_size: tuple[int, int],
) -> Tensor:
    source_h, source_w = density_map.shape[-2:]
    target_h, target_w = target_size

    if (source_h, source_w) == (target_h, target_w):
        return density_map

    resized_density = F.interpolate(
        density_map,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    scale = (source_h * source_w) / float(target_h * target_w)
    return resized_density * scale
