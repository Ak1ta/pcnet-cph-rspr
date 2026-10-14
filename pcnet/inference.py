from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .checkpoint import extract_state_dict, load_torch_checkpoint
from .config import ModelConfig
from .data import SaliencyTestDataset
from .model import PCNet


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    mixed_precision: bool = False,
) -> PCNet:
    checkpoint = load_torch_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint has no model_config and is not from PCNet-refactored."
        )
    values = dict(checkpoint["model_config"])
    values["gpu_id"] = device.index or 0
    values["mixed_precision"] = mixed_precision
    values["activation_checkpoint"] = False
    model = PCNet(ModelConfig.from_dict(values))
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def normalized_prediction(logits: torch.Tensor, shape) -> np.ndarray:
    prediction = F.interpolate(
        logits.float(), size=shape, mode="bilinear", align_corners=False
    ).sigmoid()
    prediction = prediction.cpu().numpy().squeeze()
    return (prediction - prediction.min()) / (
        prediction.max() - prediction.min() + 1e-8
    )


@torch.no_grad()
def generate_predictions(
    model: PCNet,
    dataset_root: Path,
    output_root: Path,
    image_size: int,
    device: torch.device,
    mixed_precision: bool = False,
) -> Dict[str, object]:
    rgb_root = dataset_root / "RGB"
    mask_root = dataset_root / "GT"
    thermal_root = dataset_root / "T"
    dataset = SaliencyTestDataset(
        rgb_root, mask_root, thermal_root, image_size=image_size
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.cuda.synchronize(device)
    started = time.time()
    for batch in loader:
        target_shape = tuple(batch["mask"].shape[-2:])
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            output = model(
                batch["rgb"].to(device),
                batch["thermal"].to(device),
                batch["ihn_rgb"].to(device),
                batch["ihn_thermal"].to(device),
            )
        prediction = normalized_prediction(output["saliency"], target_shape)
        destination = output_root / batch["name"][0]
        Image.fromarray((prediction * 255).round().astype(np.uint8), mode="L").save(
            destination
        )
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    return {
        "samples": len(dataset),
        "seconds": elapsed,
        "fps": len(dataset) / max(elapsed, 1e-12),
        "prediction_root": str(output_root),
    }
