"""DeepGlobe image/mask discovery, reproducible splits, and PyTorch datasets."""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SamplePair:
    image: str
    mask: str


def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def discover_pairs(dataset_path: str) -> List[SamplePair]:
    """Return sorted labeled pairs, ignoring every satellite image without a mask."""
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root}\n"
            "Set dataset.path in configs/config.yaml to an extracted dataset folder "
            "or directly to dataset_roads.zip."
        )

    if _is_zip(root):
        with zipfile.ZipFile(root) as archive:
            names = set(archive.namelist())
        images = sorted(name for name in names if name.lower().endswith("_sat.jpg"))
        pairs = [
            SamplePair(image=name, mask=name[:-8] + "_mask.png")
            for name in images
            if name[:-8] + "_mask.png" in names
        ]
    else:
        images = sorted(root.rglob("*_sat.jpg"))
        pairs = []
        for image_path in images:
            mask_path = image_path.with_name(image_path.name[:-8] + "_mask.png")
            if mask_path.is_file():
                pairs.append(
                    SamplePair(
                        image=image_path.relative_to(root).as_posix(),
                        mask=mask_path.relative_to(root).as_posix(),
                    )
                )

    if not pairs:
        raise RuntimeError(f"No labeled '*_sat.jpg' / '*_mask.png' pairs found in {root}")
    return pairs


def _pairs_fingerprint(pairs: Sequence[SamplePair]) -> str:
    payload = "\n".join(f"{p.image}\t{p.mask}" for p in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_or_load_splits(
    pairs: Sequence[SamplePair],
    split_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SamplePair]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    path = Path(split_file)
    fingerprint = _pairs_fingerprint(pairs)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        settings_match = (
            data.get("fingerprint") == fingerprint
            and data.get("seed") == seed
            and data.get("ratios") == [train_ratio, val_ratio, test_ratio]
        )
        if settings_match:
            return {
                name: [SamplePair(**item) for item in data[name]]
                for name in ("train", "val", "test")
            }

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": [train_ratio, val_ratio, test_ratio],
        "fingerprint": fingerprint,
        "counts": {name: len(items) for name, items in splits.items()},
        **{name: [asdict(item) for item in items] for name, items in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits


class PairedAugment:
    """Apply matching geometry to image/mask and image-only photometric changes."""

    def __init__(
        self,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.5,
        rotation_degrees: float = 20.0,
        brightness_contrast: float = 0.2,
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees
        self.brightness_contrast = brightness_contrast

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.vertical_flip:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        if self.brightness_contrast > 0:
            low, high = 1.0 - self.brightness_contrast, 1.0 + self.brightness_contrast
            image = ImageEnhance.Brightness(image).enhance(random.uniform(low, high))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(low, high))
        return image, mask


class DeepGlobeDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        pairs: Sequence[SamplePair],
        image_size: int = 256,
        mask_threshold: int = 128,
        augment: Optional[PairedAugment] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser()
        self.pairs = list(pairs)
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.augment = augment
        self._archive: Optional[zipfile.ZipFile] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _read(self, relative_path: str) -> bytes:
        if _is_zip(self.dataset_path):
            if self._archive is None:
                self._archive = zipfile.ZipFile(self.dataset_path)
            return self._archive.read(relative_path)
        return (self.dataset_path / relative_path).read_bytes()

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = Image.open(io.BytesIO(self._read(pair.image))).convert("RGB")
        mask = Image.open(io.BytesIO(self._read(pair.mask))).convert("L")

        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) >= self.mask_threshold).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": pair.image,
            "mask_path": pair.mask,
        }


def build_datasets(config: dict) -> Tuple[Dict[str, DeepGlobeDataset], Dict[str, List[SamplePair]]]:
    dataset_cfg = config["dataset"]
    output_dir = Path(config["paths"]["output_dir"])
    pairs = discover_pairs(dataset_cfg["path"])
    splits = create_or_load_splits(
        pairs,
        str(output_dir / "splits.json"),
        dataset_cfg["split"]["train"],
        dataset_cfg["split"]["val"],
        dataset_cfg["split"]["test"],
        config["random_seed"],
    )
    aug_cfg = config.get("augmentation", {})
    train_augment = PairedAugment(
        horizontal_flip=aug_cfg.get("horizontal_flip", 0.5),
        vertical_flip=aug_cfg.get("vertical_flip", 0.5),
        rotation_degrees=aug_cfg.get("rotation_degrees", 20),
        brightness_contrast=aug_cfg.get("brightness_contrast", 0.2),
    )
    common = {
        "dataset_path": dataset_cfg["path"],
        "image_size": dataset_cfg.get("image_size", 256),
        "mask_threshold": dataset_cfg.get("mask_threshold", 128),
    }
    datasets = {
        "train": DeepGlobeDataset(pairs=splits["train"], augment=train_augment, **common),
        "val": DeepGlobeDataset(pairs=splits["val"], **common),
        "test": DeepGlobeDataset(pairs=splits["test"], **common),
    }
    return datasets, splits
