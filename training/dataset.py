"""Carvana train dataset — verbatim from `unet_segmentation_quantised.ipynb` cell 10.

Plus the train/val split helper from cell 12 (random_split with a seeded
`torch.Generator`, so the val set is reproducible across runs).
"""

import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CarvanaTrainDataset(Dataset):
    """Carvana train dataset returning (image, mask) tuples."""

    def __init__(
        self,
        train_root: str,
        mask_root: str,
        scale: float = 0.5,
        max_samples: int = 2000,
    ) -> None:
        if scale <= 0 or scale > 1:
            raise ValueError(f"Scale must be in (0, 1], got {scale}")

        self.train_root = train_root
        self.mask_root = mask_root
        self.scale = scale

        allowed_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files: List[str] = []
        for name in os.listdir(train_root):
            path = os.path.join(train_root, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(name)
            if ext in allowed_ext:
                image_files.append(name)
        image_files.sort()

        paired: List[Tuple[str, str]] = []
        for img_name in image_files:
            stem, _ = os.path.splitext(img_name)
            mask_name = f"{stem}_mask.gif"
            mask_path = os.path.join(mask_root, mask_name)
            img_path = os.path.join(train_root, img_name)
            if os.path.isfile(mask_path):
                paired.append((img_path, mask_path))

        if not paired:
            raise RuntimeError(
                f"No image/mask pairs found in train_root='{train_root}' and mask_root='{mask_root}'."
            )

        self.pairs = paired[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def _preprocess_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            new_w = int(self.scale * w)
            new_h = int(self.scale * h)
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32)
        arr = arr.transpose((2, 0, 1))
        if arr.max() > 1.0:
            arr = arr / 255.0
        return torch.from_numpy(arr.astype(np.float32))

    def _preprocess_mask(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L")
            w, h = img.size
            new_w = int(self.scale * w)
            new_h = int(self.scale * h)
            img = img.resize((new_w, new_h), resample=Image.NEAREST)
            arr = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy((arr > 0).astype(np.int64))

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        return self._preprocess_image(img_path), self._preprocess_mask(mask_path)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_splits(dataset: Dataset, val_percent: float = 0.2, seed: int = SEED):
    """Match cell 12: random_split with a seeded torch.Generator."""
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=gen)
