"""Render sparsity benchmark plots from results/full_sparsity.json.

Outputs to results/:
  fig_time_vs_method.png      — bar chart of avg ms/image per method, faceted by batch
  fig_dice_vs_method.png      — bar chart of mean Dice deviation from baseline
  fig_speedup_vs_sparsity.png — line plot speedup vs achieved sparsity (magnitude sweep)
  summary.csv                 — one row per (method, batch_size)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
JSON_PATH = RESULTS / "full_sparsity.json"


def parse_key(key: str) -> tuple[str, int]:
    """'magnitude@0.5@bs=4' -> ('magnitude@0.5', 4); 'baseline_fp16@bs=2' -> ('baseline_fp16', 2)."""
    parts = key.split("@bs=")
    if len(parts) != 2:
        raise ValueError(f"unexpected key: {key}")
    return parts[0], int(parts[1])


def main() -> int:
    if not JSON_PATH.exists():
        print(f"missing {JSON_PATH} — run scripts/run_full.py first")
        return 1
    raw = json.loads(JSON_PATH.read_text())

    rows = []
    for key, payload in raw.items():
        method, bs = parse_key(key)
        b = payload["bench"]
        sp = payload["sparsity"]["sparsity"]
        rows.append({
            "method": method,
            "batch_size": bs,
            "ms_per_img": b["avg_time_per_image_ms"],
            "dice": b["mean_dice"],
            "sparsity": sp,
        })
    rows.sort(key=lambda r: (r["batch_size"], r["method"]))

    # CSV
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "batch_size", "sparsity", "ms_per_img", "dice"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"-> {RESULTS / 'summary.csv'}")

    # Group rows
    by_bs = defaultdict(list)
    for r in rows:
        by_bs[r["batch_size"]].append(r)

    # ---- fig 1: time vs method, faceted by batch ----
    fig, ax = plt.subplots(figsize=(11, 5))
    methods_order = sorted({r["method"] for r in rows},
                           key=lambda m: ("0" if m == "baseline_fp16" else m))
    width = 0.18
    x = list(range(len(methods_order)))
    for i, bs in enumerate(sorted(by_bs)):
        vals = []
        for m in methods_order:
            hit = next((r["ms_per_img"] for r in by_bs[bs] if r["method"] == m), None)
            vals.append(hit if hit is not None else 0)
        ax.bar([xi + i * width for xi in x], vals, width=width, label=f"bs={bs}")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(methods_order, rotation=20, ha="right")
    ax.set_ylabel("avg time per image (ms)")
    ax.set_title("UNet inference time per image — sparsity methods on A4000")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = RESULTS / "fig_time_vs_method.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"-> {out1}")
    plt.close(fig)

    # ---- fig 2: Dice deviation from baseline ----
    fig, ax = plt.subplots(figsize=(11, 5))
    baseline = {bs: next(r["dice"] for r in by_bs[bs] if r["method"] == "baseline_fp16")
                for bs in by_bs}
    for i, bs in enumerate(sorted(by_bs)):
        vals = []
        for m in methods_order:
            hit = next((r["dice"] for r in by_bs[bs] if r["method"] == m), None)
            vals.append((hit - baseline[bs]) if hit is not None else 0)
        ax.bar([xi + i * width for xi in x], vals, width=width, label=f"bs={bs}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(methods_order, rotation=20, ha="right")
    ax.set_ylabel("Δ Dice vs. baseline_fp16")
    ax.set_title("Quality degradation from sparsification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = RESULTS / "fig_dice_vs_method.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"-> {out2}")
    plt.close(fig)

    # ---- fig 3: speedup vs sparsity for magnitude sweep (bs=8) ----
    bs_focus = 8 if 8 in by_bs else sorted(by_bs)[0]
    base_time = next(r["ms_per_img"] for r in by_bs[bs_focus] if r["method"] == "baseline_fp16")
    mag_rows = sorted(
        (r for r in by_bs[bs_focus] if r["method"].startswith("magnitude@")),
        key=lambda r: r["sparsity"],
    )
    if mag_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [r["sparsity"] for r in mag_rows]
        ys = [base_time / r["ms_per_img"] for r in mag_rows]
        ax.plot(xs, ys, marker="o")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(f"global sparsity (achieved)")
        ax.set_ylabel("speedup vs baseline_fp16")
        ax.set_title(f"Magnitude pruning speedup at bs={bs_focus}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out3 = RESULTS / "fig_speedup_vs_sparsity.png"
        fig.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"-> {out3}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
