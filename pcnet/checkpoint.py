from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import torch


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json_dump(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(str(temporary), str(destination))


def atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(destination))


def load_torch_checkpoint(path, map_location="cpu", weights_only=False):
    """Load across PyTorch versions whose weights_only default differs."""

    try:
        return torch.load(
            str(path), map_location=map_location, weights_only=weights_only
        )
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(str(path), map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    result = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        result[key[7:] if key.startswith("module.") else key] = value
    if not result:
        raise ValueError("Checkpoint contains no model tensors.")
    return result
