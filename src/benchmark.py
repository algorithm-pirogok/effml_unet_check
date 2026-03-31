import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm


def run_inference_benchmark(
    model,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    mixed_precision: bool = False,
    num_classes: int = 2,
) -> dict:
    """Run inference over a few batches and measure speed + quality.

    The primary quality metric is foreground BinaryF1Score (equivalent to
    Dice for the positive class) computed using torchmetrics. For backward
    compatibility, the returned dict still uses the key `mean_dice`.
    """
    model.to(device)
    model.eval()

    cuda_available = device.type == "cuda" and torch.cuda.is_available()

    # Sum of *model forward* time over all measured batches (seconds).
    # DataLoader iteration and warmup are intentionally excluded.
    total_model_time = 0.0
    total_batches = 0
    batch_size = None

    # Metric accumulator over all batches (foreground class only)
    f1_metric = BinaryF1Score().to(device if cuda_available else "cpu")

    # Warmup run (not timed) if possible
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device, non_blocking=cuda_available)
            masks = masks.to(device, non_blocking=cuda_available)
            if mixed_precision and cuda_available:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(images)
            else:
                _ = model(images)
            break

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(dataloader, total=num_batches, desc="Inference", leave=False)):
            if i >= num_batches:
                break

            images = images.to(device, non_blocking=cuda_available)
            masks = masks.to(device, non_blocking=cuda_available)

            if batch_size is None:
                batch_size = images.shape[0]

            if cuda_available:
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            if mixed_precision and cuda_available:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            if cuda_available:
                torch.cuda.synchronize(device)
            end = time.perf_counter()

            elapsed = end - start
            total_model_time += elapsed

            # logits expected shape (N, C, H, W); convert to predicted mask
            if logits.dim() == 4:
                preds = torch.argmax(logits, dim=1)
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Convert multi-class labels to binary: foreground (class > 0) vs background (0)
            preds_bin = (preds > 0).long().flatten()
            masks_bin = (masks > 0).long().flatten()

            # Update F1/Dice metric (computed over all batches)
            f1_metric.update(preds_bin.to(f1_metric.device), masks_bin.to(f1_metric.device))

            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No batches processed in run_inference_benchmark")

    if batch_size is None:
        batch_size = 0

    # Total number of images that went through the model.
    num_images = total_batches * batch_size

    # Average model time per batch and per image (milliseconds).
    avg_batch_time_sec = total_model_time / total_batches
    avg_time_per_image_ms = (avg_batch_time_sec / max(batch_size, 1)) * 1000.0

    # Foreground F1 (equivalent to Dice for positive class in binary case)
    mean_dice = float(f1_metric.compute().item())

    return {
        # Pure model eval time (sum over all measured batches)
        "total_model_time_sec": total_model_time,
        "avg_batch_time_sec": avg_batch_time_sec,
        "avg_time_per_image_ms": avg_time_per_image_ms,
        # Throughput information
        "num_images": num_images,
        "num_batches": total_batches,
        "batch_size": batch_size,
        # Quality metric
        "mean_dice": mean_dice,
    }
