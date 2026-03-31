import os
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CarvanaTrainDataset(Dataset):
    """Carvana train dataset that returns (image, mask) pairs.

    Expects two sibling folders:
    - `train_root`: images (e.g. `abc.jpg`)
    - `mask_root`: corresponding masks (e.g. `abc_mask.gif`)

    Both image and mask are resized with the same scale factor and the
    preprocessing mirrors `BasicDataset.preprocess` from `milesial/Pytorch-UNet`.
    """

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

        # Collect jpg / jpeg / png files (non-recursive), sorted for determinism.
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

        # Keep only files that have a matching mask with `_mask.gif` suffix.
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

        self.pairs = paired[: max_samples]

        # Infer target size from the first image and the scale factor.
        first_img_path, _ = self.pairs[0]
        with Image.open(first_img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
        new_w = int(self.scale * w)
        new_h = int(self.scale * h)
        self.img_size: Tuple[int, int] = (new_h, new_w)  # (H, W)

    def __len__(self) -> int:
        return len(self.pairs)

    def _preprocess_image(self, path: str) -> torch.Tensor:
        """Preprocess a single RGB image similar to BasicDataset.preprocess."""
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            # Resize using scale factor and bicubic interpolation
            new_w = int(self.scale * w)
            new_h = int(self.scale * h)
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)

            img_arr = np.asarray(img, dtype=np.float32)

        # HWC -> CHW
        img_arr = img_arr.transpose((2, 0, 1))

        # Normalize to [0, 1] if needed
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

        tensor = torch.from_numpy(img_arr.astype(np.float32))
        return tensor

    def _preprocess_mask(self, path: str) -> torch.Tensor:
        """Preprocess a single mask as a binary (0/1) tensor with shape (H, W)."""
        with Image.open(path) as img:
            # Carvana masks are single-channel gifs; convert to L just in case.
            img = img.convert("L")
            w, h = img.size

            new_w = int(self.scale * w)
            new_h = int(self.scale * h)
            img = img.resize((new_w, new_h), resample=Image.NEAREST)

            mask_arr = np.asarray(img, dtype=np.uint8)

        # Threshold to {0, 1}. Carvana masks are usually 0/255.
        mask_bin = (mask_arr > 0).astype(np.int64)
        tensor = torch.from_numpy(mask_bin)
        return tensor

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = self._preprocess_image(img_path)
        mask = self._preprocess_mask(mask_path)
        return image, mask


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
