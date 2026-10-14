from __future__ import annotations

import argparse
from pathlib import Path

from pcnet.evaluation import (
    DATASETS,
    evaluate_directory,
    predictions_complete,
    validate_dataset,
    write_benchmark_summary,
)
from pcnet.paths import DEFAULT_DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the common five-dataset PCNet benchmark and report exactly "
            "em, sm, adpfm, wfm, mae."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--prediction-root",
        type=Path,
        help="Evaluate existing maps stored in one subfolder per dataset key.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prediction-output-root", type=Path, default=None)
    parser.add_argument("--metrics-output-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-nonstandard-dataset-size", action="store_true")
    args = parser.parse_args()
    if args.prediction_root is not None and args.force:
        parser.error("--force is only valid with --checkpoint.")
    return args


def _default_run_root(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent / checkpoint.stem


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    )
    external_predictions = (
        args.prediction_root.expanduser().resolve()
        if args.prediction_root is not None
        else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    if external_predictions is not None and not external_predictions.is_dir():
        raise FileNotFoundError(
            "Prediction root does not exist: {}".format(external_predictions)
        )

    if checkpoint is not None:
        run_root = _default_run_root(checkpoint)
        prediction_root = (
            args.prediction_output_root.expanduser().resolve()
            if args.prediction_output_root is not None
            else run_root / "predictions"
        )
        metrics_root = (
            args.metrics_output_root.expanduser().resolve()
            if args.metrics_output_root is not None
            else run_root / "metrics"
        )
        import torch

        from pcnet.inference import generate_predictions, load_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when benchmark inference is requested.")
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        model = load_model(checkpoint, device, args.mixed_precision)
    else:
        prediction_root = external_predictions
        metrics_root = (
            args.metrics_output_root.expanduser().resolve()
            if args.metrics_output_root is not None
            else external_predictions.parent / "metrics"
        )
        model = device = None

    results = {}
    had_failure = False
    for key in args.datasets:
        spec = DATASETS[key]
        result = {"status": "pending"}
        results[key] = result
        try:
            counts = validate_dataset(
                data_root,
                spec,
                require_modalities=checkpoint is not None,
                allow_nonstandard_size=args.allow_nonstandard_dataset_size,
            )
            result["samples"] = counts["GT"]
            dataset_root = data_root / Path(spec.relative_root)
            current_predictions = prediction_root / key
            if checkpoint is not None and (
                args.force
                or not predictions_complete(current_predictions, dataset_root / "GT")
            ):
                inference_result = generate_predictions(
                    model,
                    dataset_root,
                    current_predictions,
                    args.image_size,
                    device,
                    args.mixed_precision,
                )
                result["inference"] = inference_result
            metrics = evaluate_directory(
                current_predictions, dataset_root / "GT", args.num_bits
            )
            result.update(
                status="completed",
                prediction_root=str(current_predictions),
                metrics=metrics,
            )
            print("{}: {}".format(key, metrics), flush=True)
        except Exception as error:
            result.update(status="failed", message=str(error))
            had_failure = True
            print("{} FAILED: {}".format(key, error), flush=True)
            if args.fail_fast:
                write_benchmark_summary(
                    metrics_root,
                    results,
                    {"checkpoint": str(checkpoint) if checkpoint else None},
                )
                break
        write_benchmark_summary(
            metrics_root,
            results,
            {
                "checkpoint": str(checkpoint) if checkpoint else None,
                "prediction_root": str(prediction_root),
                "data_root": str(data_root),
            },
        )
    print("Summary: {}".format(metrics_root / "summary.csv"))
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
