from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .attributes import labels_for_name, load_attribute_table, validate_sample_coverage


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _files_by_stem(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(root))
    result: Dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError("Duplicate image stem in {}: {}".format(root, path.stem))
        result[key] = path
    return result


def matched_modalities(
    rgb_root: Path, mask_root: Path, thermal_root: Path
) -> Tuple[Tuple[Path, Path, Path], ...]:
    rgb = _files_by_stem(rgb_root)
    masks = _files_by_stem(mask_root)
    thermal = _files_by_stem(thermal_root)
    if not rgb:
        raise RuntimeError("No RGB images found in {}".format(rgb_root))
    if set(rgb) != set(masks) or set(rgb) != set(thermal):
        raise ValueError(
            "RGB/GT/T filename sets differ: RGB={}, GT={}, T={}.".format(
                len(rgb), len(masks), len(thermal)
            )
        )
    return tuple((rgb[key], masks[key], thermal[key]) for key in sorted(rgb))


def random_flip(image: Image.Image, mask: Image.Image, thermal: Image.Image):
    if random.randint(0, 1) == 1:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        thermal = thermal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image, mask, thermal


def random_crop(image: Image.Image, mask: Image.Image, thermal: Image.Image):
    border = 30
    width, height = image.size
    crop_width = np.random.randint(width - border, width)
    crop_height = np.random.randint(height - border, height)
    region = (
        (width - crop_width) >> 1,
        (height - crop_height) >> 1,
        (width + crop_width) >> 1,
        (height + crop_height) >> 1,
    )
    return image.crop(region), mask.crop(region), thermal.crop(region)


def random_rotation(image: Image.Image, mask: Image.Image, thermal: Image.Image):
    if random.random() > 0.8:
        angle = int(np.random.randint(-15, 15))
        image = image.rotate(angle, Image.Resampling.BICUBIC)
        mask = mask.rotate(angle, Image.Resampling.BICUBIC)
        thermal = thermal.rotate(angle, Image.Resampling.BICUBIC)
    return image, mask, thermal


def color_enhance(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.randint(5, 15) / 10.0)
    image = ImageEnhance.Contrast(image).enhance(random.randint(5, 15) / 10.0)
    image = ImageEnhance.Color(image).enhance(random.randint(0, 20) / 10.0)
    return ImageEnhance.Sharpness(image).enhance(random.randint(0, 30) / 10.0)


def salt_and_pepper(mask: Image.Image) -> Image.Image:
    values = np.array(mask)
    count = int(0.0015 * values.shape[0] * values.shape[1])
    for _ in range(count):
        row = random.randint(0, values.shape[0] - 1)
        column = random.randint(0, values.shape[1] - 1)
        values[row, column] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(values)


def boundary_from_mask(mask: Image.Image) -> Image.Image:
    """Two-sided 3x3 morphological boundary derived from saliency GT."""

    foreground = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    height, width = foreground.shape
    padded = np.pad(foreground, 1, mode="edge")
    dilated = np.zeros_like(foreground)
    eroded = np.ones_like(foreground)
    for row_offset in range(3):
        for column_offset in range(3):
            shifted = padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            dilated |= shifted
            eroded &= shifted
    return Image.fromarray((dilated & ~eroded).astype(np.uint8) * 255, mode="L")


def _rgb(path: Path) -> Image.Image:
    with path.open("rb") as handle:
        return Image.open(handle).convert("RGB")


def _gray(path: Path) -> Image.Image:
    with path.open("rb") as handle:
        return Image.open(handle).convert("L")


class SaliencyTrainDataset(Dataset):
    """One augmentation pipeline shared by all four experiments."""

    def __init__(
        self,
        rgb_root: Path,
        mask_root: Path,
        thermal_root: Path,
        image_size: int,
        attribute_file: Optional[Path] = None,
        include_boundary: bool = False,
    ):
        self.samples = matched_modalities(rgb_root, mask_root, thermal_root)
        self.image_size = image_size
        self.include_boundary = include_boundary
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.mask_resize = transforms.Resize((image_size, image_size))
        self.mask_transform = transforms.Compose(
            [self.mask_resize, transforms.ToTensor()]
        )
        self.boundary_transform = transforms.ToTensor()
        self.thermal_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )
        self.ihn_resize = transforms.Resize((128, 128))
        self.attribute_mapping: Optional[Mapping[str, Sequence[float]]] = None
        if attribute_file is not None:
            self.attribute_mapping, _ = load_attribute_table(str(attribute_file))
            validate_sample_coverage(
                self.attribute_mapping, (sample[0].name for sample in self.samples)
            )

    @property
    def stems(self) -> Tuple[str, ...]:
        return tuple(sample[0].stem.casefold() for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rgb_path, mask_path, thermal_path = self.samples[index]
        image = _rgb(rgb_path)
        mask = _gray(mask_path)
        thermal = _rgb(thermal_path)
        image, mask, thermal = random_flip(image, mask, thermal)
        image, mask, thermal = random_crop(image, mask, thermal)
        image, mask, thermal = random_rotation(image, mask, thermal)

        # Build the clean boundary at the same resolution as the model output.
        boundary = (
            boundary_from_mask(self.mask_resize(mask))
            if self.include_boundary
            else None
        )
        image = color_enhance(image)
        mask = salt_and_pepper(mask)

        image_tensor = self.rgb_transform(image)
        mask_tensor = self.mask_transform(mask)
        thermal_tensor = self.thermal_transform(thermal)
        sample = {
            "rgb": image_tensor,
            "mask": mask_tensor,
            "thermal": thermal_tensor,
            "ihn_rgb": self.ihn_resize(image_tensor),
            "ihn_thermal": self.ihn_resize(thermal_tensor),
            "name": rgb_path.name,
        }
        if boundary is not None:
            sample["boundary"] = self.boundary_transform(boundary)
        if self.attribute_mapping is not None:
            sample["attributes"] = labels_for_name(
                self.attribute_mapping, rgb_path.name
            )
        return sample


class SaliencyTestDataset(Dataset):
    def __init__(
        self, rgb_root: Path, mask_root: Path, thermal_root: Path, image_size: int
    ):
        self.samples = matched_modalities(rgb_root, mask_root, thermal_root)
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.thermal_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )
        self.mask_transform = transforms.ToTensor()
        self.ihn_resize = transforms.Resize((128, 128))

    @property
    def stems(self) -> Tuple[str, ...]:
        return tuple(sample[0].stem.casefold() for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rgb_path, mask_path, thermal_path = self.samples[index]
        image_tensor = self.rgb_transform(_rgb(rgb_path))
        thermal_tensor = self.thermal_transform(_rgb(thermal_path))
        return {
            "rgb": image_tensor,
            "mask": self.mask_transform(_gray(mask_path)),
            "thermal": thermal_tensor,
            "ihn_rgb": self.ihn_resize(image_tensor),
            "ihn_thermal": self.ihn_resize(thermal_tensor),
            "name": rgb_path.stem + ".png",
        }
