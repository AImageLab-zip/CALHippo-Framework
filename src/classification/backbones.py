from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torchvision import models

from src.classification.data_classes import DeepFeatureExtractorModel, FeatureSpec

DEFAULT_FEATURE_ENCODER_DIR = Path(
    os.environ.get(
        "NEURO_BRAIN_FEATURE_ENCODER_DIR",
        "data/models/classification/feature_encoder",
    )
)
UNI2H_ACCESS_URL = "https://huggingface.co/MahmoodLab/UNI2-h"


def load_resnet18_backbone() -> DeepFeatureExtractorModel:
    """Load a frozen ResNet18 feature extractor."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = DEFAULT_FEATURE_ENCODER_DIR / "resnet18"
    model_dir.mkdir(parents=True, exist_ok=True)

    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    state_dict = torch.hub.load_state_dict_from_url(
        weights.url,
        model_dir=str(model_dir),
        progress=True,
        check_hash=True,
    )
    resnet = models.resnet18(weights=None)
    resnet.load_state_dict(state_dict)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor.to(device)
    feature_extractor.eval()

    return DeepFeatureExtractorModel(
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )


def load_uni2h_backbone() -> DeepFeatureExtractorModel:
    """Load the UNI2-h pathology backbone from Hugging Face."""

    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = DEFAULT_FEATURE_ENCODER_DIR / "uni2h"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading UNI2-h model through timm; local Hugging Face cache is used "
        "when available."
    )

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    try:
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            cache_dir=model_dir,
            **timm_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "UNI2-h is gated and cannot be shared openly. Request access at "
            f"{UNI2H_ACCESS_URL}. After access is approved, authenticate with "
            "`hf auth login` or set `HF_TOKEN`, then rerun classification."
        ) from exc
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.to(device)
    model.eval()

    logger.info(f"UNI2-h model loaded successfully on {device}. Embedding dim: 1536.")

    return DeepFeatureExtractorModel(
        feature_extractor=model,
        preprocess=transform,
        device=device,
    )


BACKBONE_REGISTRY = {
    "resnet18": load_resnet18_backbone,
    "uni2h": load_uni2h_backbone,
}


def load_backbone(
    backbone_name: str,
    feature_spec: FeatureSpec,
) -> DeepFeatureExtractorModel | None:
    """Load a deep backbone only when the selected pipeline needs embeddings."""

    if not feature_spec.requires_deep_features:
        return None

    # Load model function from registry
    load_backbone_func = BACKBONE_REGISTRY.get(backbone_name)
    if load_backbone_func is None:
        raise ValueError(
            f"Unknown feature model: {backbone_name}. "
            f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )

    # Initialize backbone
    deep_feature_extractor_model = load_backbone_func()

    return deep_feature_extractor_model
