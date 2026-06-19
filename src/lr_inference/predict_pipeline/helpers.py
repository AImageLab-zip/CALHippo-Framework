from __future__ import annotations

from pathlib import Path

from loguru import logger


def find_yaml(run_dir: Path) -> Path:
    """Find the single YAML config inside *run_dir*."""
    yamls = list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml"))
    if len(yamls) == 0:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    if len(yamls) > 1:
        logger.warning(
            f"Multiple YAML files found in {run_dir}: {yamls}. "
            f"Using the first one: {yamls[0]}"
        )
    return yamls[0]


def find_weights(run_dir: Path) -> Path:
    """Find the model checkpoint in *run_dir*."""
    for pattern in ("final_density_model.pth", "*.pth"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .pth model weights found in {run_dir}")


def resolve_runtime_class_list(
    class_list: list[str],
    num_classes: int,
    channel_to_predict: int | None,
) -> list[str]:
    """Return runtime class labels that match the model output channels."""
    resolved = list(class_list) if class_list else []

    if num_classes == 1:
        if channel_to_predict is not None and 0 <= channel_to_predict < len(resolved):
            return [resolved[channel_to_predict]]
        if resolved:
            return [resolved[0]]
        return ["Class 0"]

    if len(resolved) < num_classes:
        resolved.extend(f"Class {i}" for i in range(len(resolved), num_classes))
    return resolved[:num_classes]
