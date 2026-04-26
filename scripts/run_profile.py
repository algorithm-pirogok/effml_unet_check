"""PyTorch Profiler traces for the sparsity pipeline.

For each method (baseline, magnitude@0.5, 2:4) profiles a few inference
batches with full CUDA + CPU activity, then exports:

  results/profile_<method>.trace.json — chrome://tracing-readable
  results/profile_<method>.summary.txt — top kernels by CUDA time + memory

Run on the same machine that runs `run_full.py`. Single batch size (default 8).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, random_split

from src.carvana_dataset import CarvanaTrainDataset
from src.sparsity import (
    apply_2to4_semi_structured,
    apply_unstructured_magnitude,
)
from src.unet import make_full_unet

WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
CARVANA_TRAIN = Path(os.environ.get("CARVANA_TRAIN", str(ROOT / "data" / "carvana" / "train")))
CARVANA_MASKS = Path(os.environ.get("CARVANA_MASKS", str(ROOT / "data" / "carvana" / "train_masks")))
RESULTS = ROOT / "results"
BATCH_SIZE = int(os.environ.get("PROFILE_BS", "8"))
SEED = 42


def build_model(device: torch.device) -> torch.nn.Module:
    m = make_full_unet(n_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    m.load_state_dict(state, strict=True)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_val_loader(scale: float = 0.5):
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
    return DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def profile_method(name: str, model_factory, loader, device) -> None:
    model = model_factory()
    print(f"\n=== profiling {name} ===")
    print(f"   params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # warmup outside the profiler so we time steady-state.
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(images)
            break
    torch.cuda.synchronize(device)

    # 2 wait + 1 warmup + 4 active = profile 4 active batches
    sched = schedule(wait=2, warmup=1, active=4)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if i >= 7:  # wait+warmup+active = 7
                    break
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(images)
                torch.cuda.synchronize(device)
                prof.step()

    trace_path = RESULTS / f"profile_{name}.trace.json"
    summary_path = RESULTS / f"profile_{name}.summary.txt"
    prof.export_chrome_trace(str(trace_path))
    print(f"   -> {trace_path}")

    table = prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=25, max_src_column_width=60
    )
    summary_path.write_text(table)
    print(f"   -> {summary_path}")
    print(table[:1500])
    del model
    torch.cuda.empty_cache()


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  torch: {torch.__version__}  bs: {BATCH_SIZE}")
    RESULTS.mkdir(exist_ok=True)
    loader = build_val_loader()

    profile_method("baseline_fp16", lambda: build_model(device), loader, device)
    profile_method("magnitude@0.5",
                   lambda: apply_unstructured_magnitude(build_model(device), 0.5),
                   loader, device)
    profile_method("2to4",
                   lambda: apply_2to4_semi_structured(build_model(device)),
                   loader, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
