from __future__ import annotations

import argparse
import platform
from pathlib import Path

from pcnet.evaluation import DATASETS, image_files, validate_dataset
from pcnet.paths import (
    DEFAULT_DATA_ROOT,
    DEFAULT_IHN_CHECKPOINT,
    DEFAULT_SWIN_CHECKPOINT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the server before launching formal PCNet runs."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--swin-checkpoint", type=Path, default=DEFAULT_SWIN_CHECKPOINT
    )
    parser.add_argument("--ihn-checkpoint", type=Path, default=DEFAULT_IHN_CHECKPOINT)
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Require UVT20K Train/Test only; skip the four external benchmarks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy
    import PIL
    import torch
    import torchvision

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("NumPy:", numpy.__version__)
    print("Pillow:", PIL.__version__)
    if not args.training_only:
        import py_sod_metrics

        print("PySODMetrics:", getattr(py_sod_metrics, "__version__", "installed"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    print("CUDA runtime:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print("GPU {}: {}".format(index, torch.cuda.get_device_name(index)))

    for checkpoint in (args.swin_checkpoint, args.ihn_checkpoint):
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Missing pretrained checkpoint: {}".format(checkpoint))
        print("Checkpoint: {} ({:.1f} MiB)".format(checkpoint, checkpoint.stat().st_size / 2**20))

    data_root = args.data_root.expanduser().resolve()
    train_root = data_root / "UVT20K" / "Train"
    train_sets = {
        name: image_files(train_root / name) for name in ("RGB", "T", "GT")
    }
    if any(len(values) != 10_000 for values in train_sets.values()):
        raise ValueError(
            "UVT20K Train requires 10,000 RGB/T/GT samples: {}".format(
                {name: len(values) for name, values in train_sets.items()}
            )
        )
    if not (set(train_sets["RGB"]) == set(train_sets["T"]) == set(train_sets["GT"])):
        raise ValueError("UVT20K Train RGB/T/GT filename sets differ.")
    if not (train_root / "Train.xlsx").is_file():
        raise FileNotFoundError("Missing CPH labels: {}".format(train_root / "Train.xlsx"))
    print("UVT20K Train: 10,000 aligned samples + Train.xlsx")
    datasets = (DATASETS["UVT20K"],) if args.training_only else DATASETS.values()
    for spec in datasets:
        counts = validate_dataset(data_root, spec, require_modalities=True)
        print("{}: {}".format(spec.key, counts))
    if args.training_only:
        print("Training environment and UVT20K Train/Test are ready.")
    else:
        print("Environment and all five datasets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
