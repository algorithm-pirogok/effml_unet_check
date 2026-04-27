"""Stand-alone cuSPARSELt 2:4 vs dense fp16 GEMM benchmark.

Why this exists:
  Inside UNet we cannot easily exploit hardware 2:4 acceleration without
  rewriting every Conv2d into im2col + sparse matmul + col2im. To still
  *demonstrate* that 2:4 sparsity *can* deliver speedup on this exact GPU,
  we benchmark the same matmul shape that occurs in the UNet bottleneck
  (1024x1024 -> 1024) once dense and once 2:4 via
  `torch.sparse.to_sparse_semi_structured` (cuSPARSELt backend on Ampere+).

Output: results/cusparselt_stand.json with per-shape {dense_ms, sparse_ms,
speedup, ok}.

Reference: https://pytorch.org/blog/accelerating-large-language-models/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

# Force the cuSPARSELt path (faster than the default CUTLASS one for fp16).
SparseSemiStructuredTensor._FORCE_CUTLASS = False

# Shapes pulled from the UNet flow at scale=0.5 input ~640x428 -> downsampled
# (m, k, n) — y = x @ W^T + b   for nn.Linear(in=k, out=n) over batch of m
# We mirror the spatial flatten of conv at deepest stages.
SHAPES = [
    # Bottleneck-like: deep features, 1024 channels
    {"name": "bottleneck_1024",   "m": 4096, "k": 1024, "n": 1024},
    # Mid-decoder: 512 channels
    {"name": "mid_decoder_512",   "m": 8192, "k": 512,  "n": 512},
    # Pointwise (OutConv) projection: 64 -> n_classes (small, sparsity overhead may dominate)
    {"name": "pointwise_64",      "m": 65536, "k": 64,  "n": 32},
    # Larger problem to see the asymptotic speedup
    {"name": "large_2048",        "m": 4096, "k": 2048, "n": 2048},
    # LLM-shape: this is where cuSPARSELt is documented to win.
    {"name": "llm_4096_4096",     "m": 4096, "k": 4096, "n": 4096},
    {"name": "llm_8192_8192",     "m": 8192, "k": 8192, "n": 8192},
]

WARMUP = 5
ITERS = 50


def make_2of4_weight(out_features: int, in_features: int, dtype, device) -> torch.Tensor:
    """Random fp16 weight with strict 2-of-4 pattern along `in_features` axis."""
    g = torch.Generator(device=device).manual_seed(0)
    w = torch.randn(out_features, in_features, dtype=dtype, device=device, generator=g) * 0.1
    n_full = (in_features // 4) * 4
    body = w[:, :n_full].reshape(out_features, -1, 4)
    _, idx = torch.topk(body.abs(), k=2, dim=-1, largest=False)
    mask = torch.ones_like(body)
    mask.scatter_(-1, idx, 0.0)
    body.mul_(mask)
    w[:, :n_full] = body.reshape(out_features, n_full)
    return w.contiguous()


def time_matmul(fn, *args, iters: int = ITERS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.float16
    cap = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    print(f"device: {name}  cap: sm_{cap[0]}{cap[1]}  torch: {torch.__version__}")

    out: list[dict] = []

    for shape in SHAPES:
        m, k, n = shape["m"], shape["k"], shape["n"]
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = make_2of4_weight(n, k, dtype, device)

        zeros = (w == 0).float().mean().item()
        print(f"\n=== {shape['name']}  m={m} k={k} n={n}  weight zeros={zeros:.3f} ===")

        # dense
        try:
            dense_ms = time_matmul(lambda x_, w_: x_ @ w_.T, x, w)
        except Exception as e:
            print(f"  dense failed: {e}")
            dense_ms = float("nan")

        # sparse 2:4 via cuSPARSELt (or CUTLASS if FORCE_CUTLASS=True)
        sparse_ok = True
        try:
            sw = to_sparse_semi_structured(w)
            sparse_ms = time_matmul(lambda x_, sw_: torch.nn.functional.linear(x_, sw_), x, sw)
        except Exception as e:
            print(f"  sparse failed: {e}")
            sparse_ms = float("nan")
            sparse_ok = False

        speedup = dense_ms / sparse_ms if sparse_ms == sparse_ms and sparse_ms > 0 else float("nan")
        print(f"  dense  : {dense_ms:7.3f} ms")
        print(f"  sparse : {sparse_ms:7.3f} ms")
        print(f"  speedup: {speedup:.2f}x")

        out.append({
            **shape,
            "dense_ms": dense_ms,
            "sparse_ms": sparse_ms,
            "speedup": speedup,
            "ok": sparse_ok,
        })

    out_path = ROOT / "results" / "cusparselt_stand.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
