"""Inference benchmark runner — same shape as Anton's `unet_benchmarks.json`.

Per-config returns:
  total_model_time_sec, avg_batch_time_sec, avg_time_per_image_ms,
  num_images, num_batches, batch_size, mean_dice
+ method-specific extras (sparsity dict, amount, etc.)
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm


def run_inference_benchmark(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
    mixed_precision: bool = True,
) -> dict:
    """Time `num_batches` of inference and compute Dice (BinaryF1) over them."""
    model.to(device).eval()
    cuda = device.type == "cuda" and torch.cuda.is_available()
    metric = BinaryF1Score().to(device)

    # warmup (untimed)
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=cuda)
            if mixed_precision and cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(images)
            else:
                _ = model(images)
            break

    total_time = 0.0
    seen_batches = 0
    batch_size = None

    with torch.no_grad():
        for i, (images, masks) in enumerate(
            tqdm(dataloader, total=min(num_batches, len(dataloader)),
                 desc="inference", leave=False)
        ):
            if i >= num_batches:
                break
            images = images.to(device, non_blocking=cuda)
            masks = masks.to(device, non_blocking=cuda)
            if batch_size is None:
                batch_size = images.shape[0]

            if cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if mixed_precision and cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)
            if cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time += t1 - t0

            if logits.dim() != 4:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            preds = torch.argmax(logits, dim=1)
            metric.update((preds > 0).long().flatten(),
                          (masks > 0).long().flatten())
            seen_batches += 1

    if seen_batches == 0:
        raise RuntimeError("No batches processed in run_inference_benchmark")

    avg_batch = total_time / seen_batches
    return {
        "total_model_time_sec": total_time,
        "avg_batch_time_sec": avg_batch,
        "avg_time_per_image_ms": (avg_batch / batch_size) * 1000.0,
        "num_images": seen_batches * batch_size,
        "num_batches": seen_batches,
        "batch_size": batch_size,
        "mean_dice": float(metric.compute().item()),
    }


def num_batches_for(num_images: int, batch_size: int) -> int:
    return max(1, math.ceil(num_images / batch_size))
