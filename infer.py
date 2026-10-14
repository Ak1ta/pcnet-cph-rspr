from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pcnet.evaluation import DATASETS, validate_dataset
from pcnet.inference import generate_predictions, load_model
from pcnet.paths import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions from any PCNet-refactored checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=["UVT20K"]
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--allow-nonstandard-dataset-size", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PCNet inference.")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    model = load_model(checkpoint, device, args.mixed_precision)
    if args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    elif checkpoint.parent.name == "checkpoints":
        output_root = checkpoint.parent.parent / "predictions"
    else:
        output_root = DEFAULT_OUTPUT_ROOT / "inference" / checkpoint.stem

    for key in args.datasets:
        spec = DATASETS[key]
        validate_dataset(
            data_root,
            spec,
            require_modalities=True,
            allow_nonstandard_size=args.allow_nonstandard_dataset_size,
        )
        result = generate_predictions(
            model,
            data_root / Path(spec.relative_root),
            output_root / key,
            args.image_size,
            device,
            args.mixed_precision,
        )
        print(
            "{}: {} samples, {:.3f} FPS -> {}".format(
                key, result["samples"], result["fps"], result["prediction_root"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
