from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def _weights_for(architecture: str, weights: Optional[str]):
    if weights is None or str(weights).lower() in {"none", "false"}:
        return None
    if str(weights).upper() == "DEFAULT":
        try:
            return models.get_model_weights(architecture).DEFAULT
        except Exception:
            return "DEFAULT"
    return weights


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            for idx in reversed(range(len(classifier))):
                layer = classifier[idx]
                if isinstance(layer, nn.Linear):
                    classifier[idx] = nn.Linear(layer.in_features, num_classes)
                    return model
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, num_classes))
            return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        head = model.heads.head
        model.heads.head = nn.Linear(head.in_features, num_classes)
        return model
    raise ValueError("Could not find a supported classifier head to replace.")


def build_model(
    architecture: str,
    num_classes: int,
    weights: Optional[str] = "DEFAULT",
    device: Optional[str] = None,
) -> nn.Module:
    if not hasattr(models, architecture):
        raise ValueError(f"Unknown torchvision architecture: {architecture}")
    model_fn = getattr(models, architecture)
    model = model_fn(weights=_weights_for(architecture, weights))
    model = _replace_classifier(model, num_classes)
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: Optional[str] = None,
) -> nn.Module:
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.to(map_location)


def get_module_by_path(model: nn.Module, module_path: Optional[str]) -> nn.Module:
    if module_path is None:
        if hasattr(model, "features"):
            return model.features[-1]
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise ValueError("target_layer is required for this model architecture.")

    module: nn.Module = model
    for part in module_path.split("."):
        if part == "":
            continue
        if part.lstrip("-").isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module
