"""Random tensor dataset — for smoke tests when Carvana isn't available yet."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RandomSegDataset(Dataset):
    """Generates (image, mask) tensor pairs of fixed shape; no I/O."""

    def __init__(
        self,
        n_samples: int = 64,
        image_shape: tuple[int, int, int] = (3, 320, 480),
        n_classes: int = 2,
        seed: int = 0,
    ):
        self.n = n_samples
        self.shape = image_shape
        self.n_classes = n_classes
        g = torch.Generator().manual_seed(seed)
        self._images = torch.rand((n_samples, *image_shape), generator=g)
        self._masks = (
            torch.rand((n_samples, image_shape[1], image_shape[2]), generator=g) > 0.5
        ).long()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self._images[idx], self._masks[idx]
