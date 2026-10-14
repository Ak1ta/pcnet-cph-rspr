from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


METRIC_NAMES: Tuple[str, ...] = (
    "em",
    "sm",
    "adpfm",
    "wfm",
    "mae",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relative_root: str
    expected_samples: int


DATASETS: Mapping[str, DatasetSpec] = {
    "UVT20K": DatasetSpec("UVT20K", "UVT20K/Test", 10_000),
    "UVT2000": DatasetSpec("UVT2000", "UVT2000", 2_000),
    "VT5000-Test_unalign": DatasetSpec(
        "VT5000-Test_unalign", "VT5000-Test_unalign", 2_500
    ),
    "VT1000_unalign": DatasetSpec("VT1000_unalign", "VT1000_unalign", 1_000),
    "VT821_unalign": DatasetSpec("VT821_unalign", "VT821_unalign", 821),
}


def image_files(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    result: Dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError("Duplicate image stem in {}: {}".format(root, path.stem))
        result[key] = path
    return result


def validate_dataset(
    data_root: Path,
    spec: DatasetSpec,
    require_modalities: bool,
    allow_nonstandard_size: bool = False,
) -> Dict[str, int]:
    root = data_root / Path(spec.relative_root)
    masks = image_files(root / "GT")
    counts = {"GT": len(masks)}
    if require_modalities:
        rgb = image_files(root / "RGB")
        thermal = image_files(root / "T")
        if set(rgb) != set(masks) or set(thermal) != set(masks):
            raise ValueError("RGB/GT/T filename sets differ for {}.".format(spec.key))
        counts.update(RGB=len(rgb), T=len(thermal))
    if not allow_nonstandard_size and len(masks) != spec.expected_samples:
        raise ValueError(
            "{} expects {} samples, found {}.".format(
                spec.key, spec.expected_samples, len(masks)
            )
        )
    return counts


def predictions_complete(prediction_root: Path, mask_root: Path) -> bool:
    try:
        return set(image_files(prediction_root)) == set(image_files(mask_root))
    except FileNotFoundError:
        return False


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def released_pcnet_emeasure(
    prediction: np.ndarray, target: np.ndarray
) -> float:
    """Match the continuous E-measure in PCNet's released Python evaluator.

    The released implementation repeats the same threshold-independent calculation
    255 times and takes its maximum. Computing it once is mathematically identical.
    Its unusual ``N - 1`` denominator is deliberately preserved.
    """

    raw_prediction = np.asarray(prediction)
    raw_target = np.asarray(target)
    prediction = raw_prediction.astype(np.float32)
    target = raw_target.astype(np.float32)
    if prediction.shape != target.shape:
        raise ValueError("E-measure prediction and target shapes must match.")
    if target.size <= 1:
        raise ValueError("E-measure requires at least two pixels.")
    if np.issubdtype(raw_prediction.dtype, np.integer) or prediction.max() > 1.0:
        prediction = prediction / np.float32(255.0)
    if np.issubdtype(raw_target.dtype, np.integer) or target.max() > 1.0:
        target = target / np.float32(255.0)

    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    alignment = (
        2.0
        * centered_target
        * centered_prediction
        / (
            centered_target * centered_target
            + centered_prediction * centered_prediction
            + 1e-20
        )
    )
    enhanced = ((alignment + 1.0) * (alignment + 1.0)) / 4.0
    return float(enhanced.sum() / (target.size - 1 + 1e-20))


def evaluate_directory(
    prediction_root: Path,
    mask_root: Path,
    num_bits: int = 4,
) -> Dict[str, float]:
    try:
        import py_sod_metrics
    except ImportError as error:
        raise RuntimeError(
            "pysodmetrics is required for the fixed five-metric benchmark."
        ) from error

    predictions = image_files(prediction_root)
    masks = image_files(mask_root)
    if set(predictions) != set(masks):
        raise RuntimeError(
            "Prediction/GT names differ: missing predictions={}, missing GT={}.".format(
                len(set(masks) - set(predictions)),
                len(set(predictions) - set(masks)),
            )
        )
    if not masks:
        raise RuntimeError("No matched prediction/GT images were found.")

    mae = py_sod_metrics.MAE()
    fm = py_sod_metrics.Fmeasure(beta=0.3)
    sm = py_sod_metrics.Smeasure()
    wfm = py_sod_metrics.WeightedFmeasure(beta=1)
    em_total = 0.0
    for key in sorted(masks):
        mask = _read_gray(masks[key])
        prediction = _read_gray(predictions[key])
        if prediction.shape != mask.shape:
            prediction = np.asarray(
                Image.fromarray(prediction).resize(
                    (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
                )
            )
        for recorder in (mae, fm, sm, wfm):
            recorder.step(prediction, mask)
        em_total += released_pcnet_emeasure(prediction, mask)

    fm_result = fm.get_results()["fm"]
    values = {
        "em": em_total / len(masks),
        "sm": float(sm.get_results()["sm"]),
        "adpfm": float(fm_result["adp"]),
        "wfm": float(wfm.get_results()["wfm"]),
        "mae": float(mae.get_results()["mae"]),
    }
    return {name: round(values[name], num_bits) for name in METRIC_NAMES}


def write_benchmark_summary(
    output_root: Path,
    results: Mapping[str, Mapping[str, object]],
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric_names": list(METRIC_NAMES),
        "metadata": dict(metadata or {}),
        "datasets": dict(results),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "status", "samples", *METRIC_NAMES, "message"])
        for key in results:
            result = results[key]
            metrics = result.get("metrics", {})
            writer.writerow(
                [
                    key,
                    result.get("status", ""),
                    result.get("samples", ""),
                    *[metrics.get(name, "") for name in METRIC_NAMES],
                    result.get("message", ""),
                ]
            )
