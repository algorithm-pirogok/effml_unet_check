"""Per-method GPU memory breakdown for the inference path.

Loads UNet 31M, applies each sparsification, runs one inference batch (bs=8),
captures peak/current GPU memory and a memory_stats() snapshot. Outputs a
small JSON; useful as a Q&A artifact at defense (~"how does pruning affect
peak memory?" — short answer: it doesn't, because dense kernels still
materialise the full activation tensors).
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

from src.carvana_dataset import CarvanaTrainDataset
from src.sparsity import (
    apply_2to4_semi_structured,
    apply_structured_channel_pruning,
    apply_unstructured_magnitude,
)
from src.unet import make_full_unet

WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
CARVANA_TRAIN = Path(os.environ.get("CARVANA_TRAIN", str(ROOT / "data" / "carvana" / "train")))
CARVANA_MASKS = Path(os.environ.get("CARVANA_MASKS", str(ROOT / "data" / "carvana" / "train_masks")))
BATCH = int(os.environ.get("MEM_BS", "8"))
SEED = 42


def build_model(device: torch.device):
    m = make_full_unet(n_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    m.load_state_dict(state, strict=True)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_loader():
    full = CarvanaTrainDataset(str(CARVANA_TRAIN), str(CARVANA_MASKS), scale=0.5, max_samples=2000)
    n_val = int(len(full) * 0.2)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    _, val = random_split(full, [n_train, n_val], generator=g)
    return DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)


def measure(name: str, model_fn, loader, device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    model = model_fn()
    after_load = torch.cuda.memory_allocated(device)

    # one timed batch
    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(images)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)

    out = {
        "method": name,
        "params_MB":     (after_load - base_alloc) / 1e6,
        "peak_alloc_MB":  peak / 1e6,
        "current_MB":     torch.cuda.memory_allocated(device) / 1e6,
        "active_blocks":  torch.cuda.memory_stats(device).get("active.all.peak", 0),
    }
    print(f"  {name:18s}  params={out['params_MB']:7.1f} MB   peak={out['peak_alloc_MB']:7.1f} MB")
    del model
    torch.cuda.empty_cache()
    return out


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  bs: {BATCH}")
    loader = build_loader()

    out = []
    out.append(measure("baseline_fp16", lambda: build_model(device), loader, device))
    out.append(measure("magnitude@0.5",
                       lambda: apply_unstructured_magnitude(build_model(device), 0.5),
                       loader, device))
    out.append(measure("magnitude@0.9",
                       lambda: apply_unstructured_magnitude(build_model(device), 0.9),
                       loader, device))
    out.append(measure("2to4",
                       lambda: apply_2to4_semi_structured(build_model(device)),
                       loader, device))
    out.append(measure("structured@0.5",
                       lambda: apply_structured_channel_pruning(build_model(device), 0.5),
                       loader, device))

    out_path = ROOT / "results" / "memory.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
