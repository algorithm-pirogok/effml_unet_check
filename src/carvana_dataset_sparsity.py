"""Carvana dataset loader — same preprocessing as the Anton/milesial benchmark.

Bicubic resize for images, nearest for masks, scale=0.5, normalize to [0,1],
masks thresholded to {0, 1}. Returns (image_chw_float32, mask_hw_int64) tuples.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CarvanaTrainDataset(Dataset):
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

        allowed_ext = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            n for n in os.listdir(train_root)
            if os.path.splitext(n)[1].lower() in allowed_ext
            and os.path.isfile(os.path.join(train_root, n))
        )

        paired: List[Tuple[str, str]] = []
        for name in image_files:
            stem = os.path.splitext(name)[0]
            mask_path = os.path.join(mask_root, f"{stem}_mask.gif")
            img_path = os.path.join(train_root, name)
            if os.path.isfile(mask_path):
                paired.append((img_path, mask_path))

        if not paired:
            raise RuntimeError(
                f"No image/mask pairs in train_root='{train_root}', mask_root='{mask_root}'"
            )

        self.pairs = paired[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def _preprocess_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            img = img.resize((int(self.scale * w), int(self.scale * h)),
                             resample=Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32)
        arr = arr.transpose((2, 0, 1))
        if arr.max() > 1.0:
            arr = arr / 255.0
        return torch.from_numpy(arr.astype(np.float32))

    def _preprocess_mask(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L")
            w, h = img.size
            img = img.resize((int(self.scale * w), int(self.scale * h)),
                             resample=Image.NEAREST)
            arr = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy((arr > 0).astype(np.int64))

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        return self._preprocess_image(img_path), self._preprocess_mask(mask_path)
