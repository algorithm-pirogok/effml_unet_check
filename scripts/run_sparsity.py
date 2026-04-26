"""Full sparsity benchmark on the 31M-parameter UNet + Carvana val split.

Loads Anton's `last.pth`, runs the same config matrix as the smoke test, but
on the real model and real data. JSON output matches Anton's
`unet_benchmarks.json` field set so the team table can be merged trivially.

Hard-coded paths point at beleriand's `~/effml/`. Adjust if running elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split

from src.bench import num_batches_for, run_inference_benchmark
from src.sparsity import (
    apply_2to4_semi_structured,
    apply_structured_channel_pruning,
    apply_unstructured_magnitude,
    measure_sparsity,
)
from src.unet import make_full_unet

# --- config ---
N_IMAGES = 1280  # match Anton's PDF setup
BATCH_SIZES = (2, 4, 8)  # bs=16 OOMs on a shared A4000 (~16GB)
MAGNITUDE_AMOUNTS = (0.3, 0.5, 0.7, 0.9)
STRUCTURED_AMOUNTS = (0.25, 0.5)
WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
CARVANA_TRAIN = Path(os.environ.get("CARVANA_TRAIN", str(ROOT / "data" / "carvana" / "train")))
CARVANA_MASKS = Path(os.environ.get("CARVANA_MASKS", str(ROOT / "data" / "carvana" / "train_masks")))
SEED = 42


def build_model(device: torch.device) -> torch.nn.Module:
    model = make_full_unet(n_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_val_dataset(scale: float = 0.5):
    from src.carvana_dataset import CarvanaTrainDataset
    full = CarvanaTrainDataset(
        train_root=str(CARVANA_TRAIN),
        mask_root=str(CARVANA_MASKS),
        scale=scale,
        max_samples=2000,
    )
    n_val = int(len(full) * 0.2)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val = random_split(full, [n_train, n_val], generator=g)
    return val


def loader_for(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def bench_one(model, loader, device, num_batches, *, mixed_precision=True) -> dict:
    return run_inference_benchmark(
        model, loader, device, num_batches=num_batches, mixed_precision=mixed_precision
    )


def cleanup() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")
    if device.type != "cuda":
        print("WARNING: not running on CUDA — final speedup numbers are meaningless.")

    val = build_val_dataset()
    print(f"val: {len(val)} samples")

    out: dict[str, dict] = {}

    def record(key: str, sparsity_global: dict, bench: dict) -> None:
        print(f"  -> {key}: {bench['avg_time_per_image_ms']:.2f} ms/img, dice={bench['mean_dice']:.4f}")
        out[key] = {"sparsity": sparsity_global, "bench": bench}

    def safe_run(key: str, build_fn) -> None:
        cleanup()
        try:
            m = build_fn()
            sp = measure_sparsity(m)
            record(key, sp["global"], bench_one(m, loader, device, nb))
            del m
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [SKIP] {key}: OOM ({e})")
            out[key] = {"error": "OOM"}
        finally:
            cleanup()

    for bs in BATCH_SIZES:
        loader = loader_for(val, bs)
        nb = num_batches_for(N_IMAGES, bs)

        safe_run(f"baseline_fp16@bs={bs}", lambda: build_model(device))

        for amt in MAGNITUDE_AMOUNTS:
            safe_run(
                f"magnitude@{amt:.1f}@bs={bs}",
                lambda amt=amt: apply_unstructured_magnitude(build_model(device), amount=amt),
            )

        safe_run(f"2to4@bs={bs}", lambda: apply_2to4_semi_structured(build_model(device)))

        for amt in STRUCTURED_AMOUNTS:
            safe_run(
                f"structured@{amt:.2f}@bs={bs}",
                lambda amt=amt: apply_structured_channel_pruning(build_model(device), amount=amt),
            )

        # Save partial results between batch sizes — protects against late crashes.
        partial = ROOT / "results" / "full_sparsity.partial.json"
        partial.parent.mkdir(exist_ok=True)
        partial.write_text(json.dumps(out, indent=2))

    out_path = ROOT / "results" / "full_sparsity.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{out_path}  ({len(out)} configs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
