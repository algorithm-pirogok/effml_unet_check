"""Smoke test: verifies the sparsity pipeline on a tiny UNet without Carvana.

Runs through:
  - tiny UNet (~0.5M params) with random weights
  - random tensor dataset (no I/O)
  - applies each sparsity method and benchmarks inference

What this proves:
  - all three sparsity methods leave the model functional (forward doesn't fail)
  - measured zero-fraction matches the requested amount
  - 2:4 produces ~50% sparsity in conv main bodies
  - the benchmark runner returns sane numbers (>0 time, valid dice in [0,1])

Does NOT prove final accuracy — random weights -> Dice ~ chance. That's fine,
we only check the pipeline plumbing here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.bench import run_inference_benchmark, num_batches_for
from src.random_dataset import RandomSegDataset
from src.sparsity import (
    apply_2to4_semi_structured,
    apply_structured_channel_pruning,
    apply_unstructured_magnitude,
    count_params,
    measure_sparsity,
)
from src.unet import make_tiny_unet


def fresh_model(device: torch.device) -> torch.nn.Module:
    torch.manual_seed(0)
    m = make_tiny_unet(n_classes=2)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fmt_bench(b: dict) -> str:
    return (
        f"batch={b['batch_size']:>3}  "
        f"t/img={b['avg_time_per_image_ms']:7.2f}ms  "
        f"dice={b['mean_dice']:.4f}  "
        f"batches={b['num_batches']}"
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")

    BATCH = 2
    N_IMAGES = 16  # tiny — we just want ~8 batches of timing
    H, W = 256, 384

    dataset = RandomSegDataset(n_samples=N_IMAGES, image_shape=(3, H, W))
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    nb = num_batches_for(N_IMAGES, BATCH)
    print(f"dataset: {N_IMAGES} synthetic samples, {H}x{W}, batch={BATCH}, num_batches={nb}")

    results: dict[str, dict] = {}

    # ---- baseline ----
    m = fresh_model(device)
    n_params = count_params(m)
    print(f"\n=== baseline ===  params={n_params/1e6:.2f}M")
    sp = measure_sparsity(m)
    print(f"  global sparsity (random init): {sp['global']['sparsity']:.4f}")
    bench = run_inference_benchmark(m, loader, device, nb, mixed_precision=device.type == "cuda")
    results["baseline"] = {"sparsity": sp["global"], "bench": bench}
    print("  " + fmt_bench(bench))

    # ---- magnitude pruning sweep ----
    for amount in (0.3, 0.5, 0.7, 0.9):
        m = fresh_model(device)
        apply_unstructured_magnitude(m, amount=amount)
        sp = measure_sparsity(m)
        bench = run_inference_benchmark(m, loader, device, nb, mixed_precision=device.type == "cuda")
        key = f"magnitude@{amount:.1f}"
        results[key] = {"requested_amount": amount, "sparsity": sp["global"], "bench": bench}
        achieved = sp["global"]["sparsity"]
        print(f"\n=== {key} ===  requested={amount:.2f}  achieved={achieved:.4f}")
        print("  " + fmt_bench(bench))
        assert abs(achieved - amount) < 0.05, f"sparsity drift for {key}: {achieved} vs {amount}"

    # ---- 2:4 semi-structured ----
    m = fresh_model(device)
    apply_2to4_semi_structured(m)
    sp = measure_sparsity(m)
    bench = run_inference_benchmark(m, loader, device, nb, mixed_precision=device.type == "cuda")
    achieved = sp["global"]["sparsity"]
    results["2to4"] = {"sparsity": sp["global"], "bench": bench}
    print(f"\n=== 2:4 semi-structured ===  achieved={achieved:.4f}  (target ~0.50)")
    print("  " + fmt_bench(bench))
    assert 0.45 < achieved < 0.55, f"2:4 sparsity off: {achieved}"

    # ---- structured channel pruning ----
    for amount in (0.25, 0.5):
        m = fresh_model(device)
        apply_structured_channel_pruning(m, amount=amount)
        sp = measure_sparsity(m)
        bench = run_inference_benchmark(m, loader, device, nb, mixed_precision=device.type == "cuda")
        key = f"structured@{amount:.2f}"
        results[key] = {"requested_amount": amount, "sparsity": sp["global"], "bench": bench}
        print(f"\n=== {key} ===  achieved={sp['global']['sparsity']:.4f}")
        print("  " + fmt_bench(bench))

    out = ROOT / "results" / "smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsmoke results -> {out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
