from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import random
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from .checkpoint import (
    atomic_json_dump,
    atomic_torch_save,
    jsonable,
)
from .config import (
    EXPECTED_TEST_SAMPLES,
    EXPECTED_TRAIN_SAMPLES,
    PROTOCOL_VERSION,
    REFERENCE_CLIP,
    REFERENCE_EPOCHS,
    REFERENCE_IMAGE_SIZE,
    REFERENCE_LEARNING_RATE,
    REFERENCE_BATCH_SIZE,
    REFERENCE_SEED,
    REFERENCE_WEIGHT_DECAY,
    ModelConfig,
    get_experiment,
    selection_epochs,
)
from .data import SaliencyTestDataset, SaliencyTrainDataset
from .losses import (
    AsymmetricLossMultiLabel,
    SoftDiceLoss,
    boundary_loss,
    clip_gradients,
    dilated_boundary,
    saliency_loss,
)
from .model import PCNet
from .paths import (
    DEFAULT_DATA_ROOT,
    DEFAULT_IHN_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SWIN_CHECKPOINT,
)


def build_parser(experiment: str) -> argparse.ArgumentParser:
    spec = get_experiment(experiment)
    parser = argparse.ArgumentParser(
        description="Train {} with the controlled PCNet protocol.".format(spec.label)
    )
    parser.add_argument("--epochs", type=int, default=REFERENCE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=REFERENCE_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--learning-rate", type=float, default=REFERENCE_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=REFERENCE_WEIGHT_DECAY)
    parser.add_argument("--clip", type=float, default=REFERENCE_CLIP)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")

    parser.add_argument("--attribute-loss-weight", type=float, default=0.1)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.25)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--edge-dilation", type=int, default=3)
    parser.add_argument("--edge-positive-weight-cap", type=float, default=20.0)
    parser.add_argument("--attribute-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rspr-channels", type=int, default=64)
    parser.add_argument("--rspr-grid-size", type=int, default=2)
    parser.add_argument("--rspr-confidence-gamma", type=float, default=4.0)

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-nonstandard-dataset-size", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--swin-checkpoint", type=Path, default=DEFAULT_SWIN_CHECKPOINT
    )
    parser.add_argument("--ihn-checkpoint", type=Path, default=DEFAULT_IHN_CHECKPOINT)
    return parser


def _world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be an integer.") from error


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in ("data_root", "output_root", "swin_checkpoint", "ihn_checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    uvt20k = args.data_root / "UVT20K"
    args.train_rgb_root = uvt20k / "Train" / "RGB"
    args.train_mask_root = uvt20k / "Train" / "GT"
    args.train_thermal_root = uvt20k / "Train" / "T"
    args.train_attribute_file = uvt20k / "Train" / "Train.xlsx"
    args.test_rgb_root = uvt20k / "Test" / "RGB"
    args.test_mask_root = uvt20k / "Test" / "GT"
    args.test_thermal_root = uvt20k / "Test" / "T"


def _validate_args(args: argparse.Namespace) -> None:
    spec = get_experiment(args.experiment)
    if _world_size() != 1:
        raise RuntimeError(
            "This protocol uses exactly one GPU."
        )
    if min(
        args.epochs,
        args.batch_size,
        args.image_size,
    ) < 1:
        raise ValueError(
            "epochs, batch size, and image size must be positive."
        )
    if args.num_workers < 0 or args.gpu_id < 0:
        raise ValueError("num-workers and gpu-id must be non-negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.clip <= 0:
        raise ValueError("Invalid optimizer settings.")
    if min(
        args.attribute_loss_weight,
        args.coarse_loss_weight,
        args.edge_loss_weight,
    ) < 0:
        raise ValueError("Auxiliary loss weights must be non-negative.")
    if args.edge_dilation < 1 or args.edge_dilation % 2 == 0:
        raise ValueError("edge-dilation must be a positive odd integer.")
    if args.edge_positive_weight_cap < 1:
        raise ValueError("edge-positive-weight-cap must be at least 1.")
    if args.log_every < 1:
        raise ValueError("log-every must be positive.")

    for path in (
        args.train_rgb_root,
        args.train_mask_root,
        args.train_thermal_root,
        args.test_rgb_root,
        args.test_mask_root,
        args.test_thermal_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError("Dataset directory does not exist: {}".format(path))
    if spec.challenge_perception and not args.train_attribute_file.is_file():
        raise FileNotFoundError(
            "CPH requires UVT20K/Train/Train.xlsx: {}".format(
                args.train_attribute_file
            )
        )
    for path in (args.swin_checkpoint, args.ihn_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(
                "Pretrained component does not exist: {}".format(path)
            )


def _choose_run_directory(args: argparse.Namespace) -> Path:
    run_name = args.run_name or (
        "smoke_seed_{}".format(args.seed)
        if args.smoke_test
        else "seed_{}".format(args.seed)
    )
    if Path(run_name).name != run_name:
        raise ValueError("--run-name must be one folder name, not a path.")
    return (args.output_root / args.experiment / run_name).resolve()


def _prepare_output(args: argparse.Namespace) -> Path:
    run_dir = _choose_run_directory(args)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Training refuses to reuse non-empty {}. Use another seed or run name.".format(
                run_dir
            )
        )
    for directory in (
        run_dir,
        run_dir / "checkpoints",
        run_dir / "logs",
        run_dir / "predictions",
        run_dir / "metrics",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir
    return run_dir


def _setup_logging(run_dir: Path) -> None:
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            str(run_dir / "logs" / "train.log"),
            mode="w",
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _dataset_contract(
    train_dataset: SaliencyTrainDataset,
    test_dataset: SaliencyTestDataset,
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_stems = train_dataset.stems
    test_stems = test_dataset.stems
    overlap = sorted(set(train_stems).intersection(test_stems))
    if overlap:
        raise ValueError(
            "Train/Test leakage detected ({} shared stems).".format(len(overlap))
        )
    if not args.allow_nonstandard_dataset_size and (
        len(train_stems) != EXPECTED_TRAIN_SAMPLES
        or len(test_stems) != EXPECTED_TEST_SAMPLES
    ):
        raise ValueError(
            "Formal UVT20K requires 10,000 Train and 10,000 Test samples; found {}/{}.".format(
                len(train_stems), len(test_stems)
            )
        )
    return {
        "train_samples": len(train_stems),
        "test_samples": len(test_stems),
        "train_stem_sha256": _digest(train_stems),
        "test_stem_sha256": _digest(test_stems),
        "train_test_overlap": 0,
        "attribute_file": (
            _file_fingerprint(args.train_attribute_file)
            if get_experiment(args.experiment).challenge_perception
            else None
        ),
        "boundary_supervision": (
            {
                "source": "saliency_gt",
                "operator": "two-sided 3x3 morphological gradient",
                "derived_after_geometric_augmentation": True,
                "derived_after_resize": True,
                "resolution": [args.image_size, args.image_size],
                "derived_before_salt_and_pepper_augmentation": True,
            }
            if get_experiment(args.experiment).rspr
            else None
        ),
    }


def _build_datasets(args: argparse.Namespace):
    spec = get_experiment(args.experiment)
    train_dataset = SaliencyTrainDataset(
        args.train_rgb_root,
        args.train_mask_root,
        args.train_thermal_root,
        args.image_size,
        attribute_file=(
            args.train_attribute_file if spec.challenge_perception else None
        ),
        include_boundary=spec.rspr,
    )
    test_dataset = SaliencyTestDataset(
        args.test_rgb_root,
        args.test_mask_root,
        args.test_thermal_root,
        args.image_size,
    )
    return train_dataset, test_dataset, _dataset_contract(
        train_dataset, test_dataset, args
    )


def _build_loaders(
    train_dataset,
    test_dataset,
    args,
    sampler_generator: torch.Generator,
    worker_generator: torch.Generator,
):
    sampler = RandomSampler(
        train_dataset, replacement=False, generator=sampler_generator
    )
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    train_options = {
        "dataset": train_dataset,
        "batch_sampler": batch_sampler,
        "num_workers": args.num_workers,
        "pin_memory": not args.no_pin_memory,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "generator": worker_generator,
    }
    if args.num_workers > 0:
        train_options["prefetch_factor"] = 1
    train_loader = DataLoader(**train_options)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=not args.no_pin_memory,
    )
    return train_loader, test_loader


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        experiment=args.experiment,
        seed=args.seed,
        attribute_hidden_dim=args.attribute_hidden_dim,
        dropout=args.dropout,
        rspr_channels=args.rspr_channels,
        rspr_grid_size=args.rspr_grid_size,
        rspr_confidence_gamma=args.rspr_confidence_gamma,
        activation_checkpoint=args.activation_checkpoint,
        gpu_id=args.gpu_id,
        mixed_precision=args.mixed_precision,
    )


def _to_device(batch, device: torch.device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _train_epoch(
    loader,
    model,
    optimizer,
    scaler,
    device,
    args,
    epoch,
    global_step,
    dice,
    attribute_loss_fn,
):
    spec = get_experiment(args.experiment)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    metric_names = (
        "loss",
        "saliency",
        "bce",
        "dice",
        "smoothness",
        "attribute",
        "coarse",
        "edge",
        "edge_bce",
        "edge_dice",
        "correction_abs_mean",
    )
    totals = {name: torch.zeros((), device=device) for name in metric_names}
    completed_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_step, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            outputs = model(
                batch["rgb"],
                batch["thermal"],
                batch["ihn_rgb"],
                batch["ihn_thermal"],
            )

        target = batch["mask"].float()
        primary = saliency_loss(outputs["saliency"], target, dice)
        loss = primary["total"]
        zero = loss.new_zeros(())
        attribute = coarse = edge_total = edge_bce = edge_dice = zero
        correction_abs_mean = zero

        if spec.challenge_perception:
            if outputs["attributes"] is None or "attributes" not in batch:
                raise RuntimeError("CPH logits or attribute labels are missing.")
            attribute = attribute_loss_fn(
                outputs["attributes"].float(), batch["attributes"].float()
            )
            loss = loss + args.attribute_loss_weight * attribute

        if spec.rspr:
            if outputs["edge"] is None or "boundary" not in batch:
                raise RuntimeError("RSPR edge output or boundary target is missing.")
            coarse_logits = outputs["coarse_saliency"].float()
            coarse = F.binary_cross_entropy_with_logits(coarse_logits, target)
            coarse = coarse + dice(coarse_logits, target)
            edge_target = dilated_boundary(
                batch["boundary"].float(), args.edge_dilation
            )
            edge_values = boundary_loss(
                outputs["edge"].float(),
                edge_target,
                dice,
                args.edge_positive_weight_cap,
            )
            edge_total = edge_values["total"]
            edge_bce = edge_values["bce"]
            edge_dice = edge_values["dice"]
            correction_abs_mean = (
                outputs["rspr_correction"].detach().float().abs().mean()
            )
            loss = (
                loss
                + args.coarse_loss_weight * coarse
                + args.edge_loss_weight * edge_total
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_gradients(optimizer, args.clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        completed_batches += 1
        values = {
            "loss": loss,
            "saliency": primary["total"],
            "bce": primary["bce"],
            "dice": primary["dice"],
            "smoothness": primary["smoothness"],
            "attribute": attribute,
            "coarse": coarse,
            "edge": edge_total,
            "edge_bce": edge_bce,
            "edge_dice": edge_dice,
            "correction_abs_mean": correction_abs_mean,
        }
        for name, value in values.items():
            totals[name].add_(value.detach())
        if (
            completed_batches == 1
            or completed_batches % args.log_every == 0
            or batch_step == len(loader)
        ):
            logging.info(
                "experiment=%s epoch=%d/%d batch=%d/%d optimizer_step=%d "
                "global_step=%d "
                "loss=%.6f saliency=%.6f attr=%.6f coarse=%.6f edge=%.6f lr=%.8g",
                args.experiment,
                epoch,
                args.epochs,
                batch_step,
                len(loader),
                completed_batches,
                global_step,
                loss.item(),
                primary["total"].item(),
                attribute.item(),
                coarse.item(),
                edge_total.item(),
                optimizer.param_groups[0]["lr"],
            )
        if args.smoke_test:
            break
    averages = {
        name: float((total / max(completed_batches, 1)).item())
        for name, total in totals.items()
    }
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    averages.update(
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_reserved_gib,
    )
    logging.info(
        "experiment=%s epoch=%d CUDA peak allocated=%.2f GiB reserved=%.2f GiB",
        args.experiment,
        epoch,
        peak_allocated_gib,
        peak_reserved_gib,
    )
    return averages, global_step, completed_batches


@torch.no_grad()
def evaluate_test_mae(loader, model, device, mixed_precision: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        target = batch["mask"].float()
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits = model(
                batch["rgb"],
                batch["thermal"],
                batch["ihn_rgb"],
                batch["ihn_thermal"],
            )["saliency"]
        prediction = F.interpolate(
            logits.float(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        prediction = (prediction - prediction.amin(dim=(-2, -1), keepdim=True)) / (
            prediction.amax(dim=(-2, -1), keepdim=True)
            - prediction.amin(dim=(-2, -1), keepdim=True)
            + 1e-8
        )
        normalized_target = target / (target.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        total += float((prediction - normalized_target).abs().mean().item())
        count += 1
    if count == 0:
        raise RuntimeError("Test loader is empty.")
    return total / count


def _training_args(args: argparse.Namespace) -> Dict[str, object]:
    excluded = {"run_dir"}
    return {
        key: jsonable(value)
        for key, value in vars(args).items()
        if key not in excluded
    }


def _model_artifact(model, epoch, global_step, args, role, test_mae=None):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": args.experiment,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_args": _training_args(args),
        "artifact_role": role,
        "selection_metric": "test_mae" if test_mae is not None else None,
        "test_mae": test_mae,
    }


def _reference_deviations(args) -> Dict[str, object]:
    expected = {
        "epochs": REFERENCE_EPOCHS,
        "batch_size": REFERENCE_BATCH_SIZE,
        "image_size": REFERENCE_IMAGE_SIZE,
        "num_workers": 4,
        "seed": REFERENCE_SEED,
        "learning_rate": REFERENCE_LEARNING_RATE,
        "weight_decay": REFERENCE_WEIGHT_DECAY,
        "clip": REFERENCE_CLIP,
        "mixed_precision": False,
        "activation_checkpoint": False,
    }
    return {
        key: {"actual": getattr(args, key), "reference": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }


def _run_config(
    args, model, contract, batches_per_epoch, load_report
):
    spec = get_experiment(args.experiment)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": args.experiment,
        "experiment_spec": spec.__dict__,
        "reference_protocol_deviations": _reference_deviations(args),
        "initialization": {
            "fresh": True,
            "trained_pcnet_checkpoint_loaded": False,
            "swin_checkpoint": {
                "path": str(args.swin_checkpoint),
                **_file_fingerprint(args.swin_checkpoint),
            },
            "ihn_checkpoint": {
                "path": str(args.ihn_checkpoint),
                **_file_fingerprint(args.ihn_checkpoint),
            },
            "load_report": load_report,
        },
        "data": contract,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps_per_epoch": batches_per_epoch,
            "planned_optimizer_steps": batches_per_epoch * args.epochs,
            "epochs": args.epochs,
            "scheduler": None,
            "gradient_clipping": "element-wise clamp",
            "clip": args.clip,
            "drop_last": True,
        },
        "loss": {
            "saliency": "BCEWithLogits + released SoftDice + Smoothness",
            "attribute_weight": (
                args.attribute_loss_weight if spec.challenge_perception else 0.0
            ),
            "coarse_weight": args.coarse_loss_weight if spec.rspr else 0.0,
            "edge_weight": args.edge_loss_weight if spec.rspr else 0.0,
        },
        "model_selection": {
            "dataset": "UVT20K/Test",
            "metric": "minimum MAE",
            "epochs": selection_epochs(args.epochs),
        },
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "innovations": model.count_branch_parameters(),
        },
        "model_config": model.config.to_dict(),
        "arguments": _training_args(args),
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.gpu_id),
            "world_size": _world_size(),
        },
    }


def main_for(experiment: str, argv: Optional[Sequence[str]] = None) -> int:
    get_experiment(experiment)
    args = build_parser(experiment).parse_args(argv)
    args.experiment = experiment
    _resolve_paths(args)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Swin-B + IHN PCNet training.")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError("gpu-id={} is unavailable.".format(args.gpu_id))
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    run_dir = _prepare_output(args)
    _setup_logging(run_dir)
    logging.warning(
        "TEST-SELECTION MODE: minimum UVT20K Test MAE selects best.pth at epochs %s.",
        selection_epochs(args.epochs),
    )
    if _reference_deviations(args):
        logging.warning("Reference protocol deviations: %s", _reference_deviations(args))

    set_seed(args.seed)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    worker_generator = torch.Generator().manual_seed(args.seed + 1)
    config = _model_config(args)
    model = PCNet(config)
    load_report = model.load_pretrained_components(
        str(args.swin_checkpoint), str(args.ihn_checkpoint)
    )
    logging.info("Loaded only Swin and IHN pretraining; no trained PCNet checkpoint.")
    model.to(device)

    train_dataset, test_dataset, contract = _build_datasets(args)
    train_loader, test_loader = _build_loaders(
        train_dataset,
        test_dataset,
        args,
        sampler_generator,
        worker_generator,
    )
    batches_per_epoch = len(train_loader)
    if batches_per_epoch == 0:
        raise RuntimeError(
            "Training loader is empty; the dataset must contain at least one full batch."
        )
    if (
        not args.allow_nonstandard_dataset_size
        and args.batch_size == REFERENCE_BATCH_SIZE
        and batches_per_epoch != 2_500
    ):
        raise AssertionError(
            "Expected 2,500 batches and optimizer steps per epoch, got {}.".format(
                batches_per_epoch
            )
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    global_step = 0
    best_mae = best_epoch = None
    training_history = []
    selection_history = []
    config_path = run_dir / "config.json"
    current_config = _run_config(
        args,
        model,
        contract,
        batches_per_epoch,
        load_report,
    )
    atomic_json_dump(current_config, config_path)

    logging.info(
        "Experiment=%s Train=%d Test=%d batch_size=%d "
        "batches/epoch=%d optimizer_steps/epoch=%d "
        "epochs=%d scheduler=none",
        args.experiment,
        contract["train_samples"],
        contract["test_samples"],
        args.batch_size,
        batches_per_epoch,
        batches_per_epoch,
        args.epochs,
    )
    dice = SoftDiceLoss()
    attribute_loss_fn = AsymmetricLossMultiLabel()
    official_epochs = set(selection_epochs(args.epochs))
    checkpoint_dir = run_dir / "checkpoints"
    try:
        for epoch in range(1, args.epochs + 1):
            averages, global_step, completed = _train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                device,
                args,
                epoch,
                global_step,
                dice,
                attribute_loss_fn,
            )
            if args.smoke_test:
                atomic_json_dump(
                    {
                        "status": "passed",
                        "experiment": args.experiment,
                        "completed_optimizer_steps": completed,
                        "averages": averages,
                    },
                    run_dir / "smoke_result.json",
                )
                logging.info("Smoke test passed: one real optimizer step completed.")
                return 0

            training_history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "averages": averages,
                }
            )
            atomic_json_dump(training_history, run_dir / "training_history.json")

            if epoch in official_epochs:
                test_mae = evaluate_test_mae(
                    test_loader, model, device, args.mixed_precision
                )
                selection_history.append(
                    {"epoch": epoch, "test_mae": test_mae}
                )
                atomic_json_dump(
                    selection_history, run_dir / "selection_history.json"
                )
                logging.info("epoch=%d UVT20K Test MAE=%.8f", epoch, test_mae)
                if best_mae is None or test_mae < best_mae:
                    best_mae, best_epoch = test_mae, epoch
                    atomic_torch_save(
                        _model_artifact(
                            model,
                            epoch,
                            global_step,
                            args,
                            "test_mae_selected_best",
                            test_mae,
                        ),
                        checkpoint_dir / "best.pth",
                    )
                    logging.info("Saved best.pth at epoch=%d MAE=%.8f", epoch, test_mae)

    except KeyboardInterrupt:
        logging.warning("Training interrupted before completion.")
        return 130

    atomic_torch_save(
        _model_artifact(
            model,
            args.epochs,
            global_step,
            args,
            "fixed_final_epoch",
        ),
        checkpoint_dir / "final.pth",
    )
    logging.info(
        "Training complete: experiment=%s best_epoch=%s best_mae=%s output=%s",
        args.experiment,
        best_epoch,
        best_mae,
        run_dir,
    )
    return 0
