"""Slide-friendly figures from every JSON we collected.

Output to results/slides/:
  01_quality_envelope.png    — Dice vs sparsity for magnitude pruning (no fine-tune)
  02_finetune_recovery.png   — paired pre/post Dice after 1 epoch fine-tune
  03_iterative_vs_oneshot.png — gradual ladder vs single shot at 70%
  04_asp_combo.png           — waterfall: baseline -> magnitude+ft -> +2:4+ft
  05_cusparselt_speedup.png  — sparse vs dense matmul speedup across shapes
  06_method_landscape.png    — Dice vs sparsity scatter, all methods one plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "slides"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Brand-ish palette (high contrast, avoid neon)
C_BASE = "#2c3e50"     # dark slate
C_MAG = "#3498db"      # blue
C_2TO4 = "#e67e22"     # orange
C_STRUCT = "#9b59b6"   # purple
C_RECOVER = "#27ae60"  # green
C_BAD = "#c0392b"      # red


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        print(f"missing {p}")
        return None
    return json.loads(p.read_text())


def parse_full_key(k: str) -> tuple[str, int]:
    method, bs = k.split("@bs=")
    return method, int(bs)


# ---------------------------------------------------------------------- 1
def fig_quality_envelope():
    """Dense sweep with and without fine-tune (uses sweep.json if present,
    otherwise falls back to the coarse 4-point full_sparsity.json)."""
    sweep = load("sweep.json")
    if sweep:
        sweep = sorted(sweep, key=lambda r: r["sparsity"])
        xs = [r["sparsity"] for r in sweep]
        pre = [r["pre_dice"] for r in sweep]
        post = [r["post_dice"] for r in sweep]
        baseline = sweep[0]["pre_dice"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xs, pre, marker="o", markersize=9, linewidth=2.5, color=C_BAD,
                label="pre fine-tune (just prune)")
        ax.plot(xs, post, marker="s", markersize=9, linewidth=2.5, color=C_RECOVER,
                label="post 1-epoch fine-tune (frozen encoder)")
        ax.axhline(baseline, ls="--", color=C_BASE, alpha=0.6,
                   label=f"baseline Dice {baseline:.4f}")
        ax.fill_between(xs, pre, post, where=[p < q for p, q in zip(pre, post)],
                        color=C_RECOVER, alpha=0.10, label="recovered by fine-tune")
        ax.set_xlabel("global sparsity (magnitude prune, full model)")
        ax.set_ylabel("Dice (BinaryF1)")
        ax.set_title("Quality envelope: cliff without retrain is gone after 1 epoch fine-tune")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.02, 1.0)
        ax.legend(loc="lower left")
        fig.tight_layout()
        out = OUT / "01_quality_envelope.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")
        return

    # Fallback: original 4-point version from full_sparsity.json
    full = load("full_sparsity.json")
    if not full:
        return
    rows = []
    for k, v in full.items():
        method, bs = parse_full_key(k)
        if bs != 8:
            continue
        if not method.startswith("magnitude@") and method != "baseline_fp16":
            continue
        rows.append({
            "method": method,
            "sparsity": v["sparsity"]["sparsity"],
            "dice": v["bench"]["mean_dice"],
        })
    rows.sort(key=lambda r: r["sparsity"])
    xs = [r["sparsity"] for r in rows]
    ys = [r["dice"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker="o", markersize=10, linewidth=2.5, color=C_MAG, label="magnitude pruning")
    ax.axhline(rows[0]["dice"], ls="--", color=C_BASE, alpha=0.6, label=f"baseline Dice {rows[0]['dice']:.4f}")
    ax.set_xlabel("global sparsity")
    ax.set_ylabel("Dice (BinaryF1)")
    ax.set_title("Quality envelope: UNet tolerates magnitude pruning up to ~70%")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.02, 1.0)
    for r in rows:
        ax.annotate(f"{r['dice']:.3f}", xy=(r["sparsity"], r["dice"]),
                    xytext=(8, -16), textcoords="offset points", fontsize=11)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = OUT / "01_quality_envelope.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------- 2
def fig_finetune_recovery():
    ft = load("finetune.json")
    if not ft:
        return
    labels = [r["label"] for r in ft]
    pre = [r["pre_dice"] for r in ft]
    post = [r["post_dice"] for r in ft]
    baseline = 0.9912

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, pre, width, label="pre fine-tune", color=C_BAD, alpha=0.8)
    b2 = ax.bar(x + width/2, post, width, label="post 1 epoch fine-tune", color=C_RECOVER, alpha=0.9)
    ax.axhline(baseline, ls="--", color=C_BASE, alpha=0.7, label=f"baseline Dice {baseline:.4f}")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice (BinaryF1)")
    ax.set_title("1 epoch of fine-tuning recovers (and exceeds) baseline quality")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT / "02_finetune_recovery.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------- 3
def fig_iterative_vs_oneshot():
    it = load("iterative.json")
    ft = load("finetune.json")
    if not it:
        return
    xs = [r["sparsity"] for r in it]
    ys = [r["post_dice"] for r in it]
    labels = [f"iter {r['step']}: target {int(r['target']*100)}%" for r in it]

    one_shot_70 = next((r for r in (ft or []) if r["label"] == "magnitude@0.7"), None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker="o", markersize=11, linewidth=2.5, color=C_RECOVER,
            label="iterative ladder (this work)")
    for x_, y_, lbl in zip(xs, ys, labels):
        ax.annotate(f"{y_:.4f}\n{lbl}",
                    xy=(x_, y_), xytext=(10, 12), textcoords="offset points", fontsize=10)
    if one_shot_70:
        ax.scatter([xs[-1]], [one_shot_70["post_dice"]], marker="X", s=180, color=C_BAD,
                   label=f"one-shot 70% (post fine-tune): {one_shot_70['post_dice']:.4f}", zorder=5)

    ax.set_xlabel("global sparsity (achieved)")
    ax.set_ylabel("Dice after fine-tune")
    ax.set_title("Iterative pruning beats one-shot")
    ax.set_ylim(0.985, 0.998)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT / "03_iterative_vs_oneshot.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------- 4
def fig_asp_combo():
    asp = load("asp_combo.json")
    if not asp:
        return
    labels = ["baseline", "magnitude\n50% + ft", "+ 2:4 mask\n+ ft"]
    values = [asp["baseline_decoder"]] + [s["post_tune_dice"] for s in asp["steps"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [C_BASE, C_MAG, C_2TO4]
    bars = ax.bar(x, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
    for rect, v in zip(bars, values):
        ax.annotate(f"{v:.4f}", xy=(rect.get_x() + rect.get_width()/2, v),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=12)

    ax.axhline(asp["baseline_decoder"], ls="--", color=C_BASE, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.985, 0.998)
    ax.set_ylabel("Dice after stage")
    ax.set_title("Full NVIDIA ASP recipe: magnitude → 2:4 → fine-tune")
    fig.tight_layout()
    out = OUT / "04_asp_combo.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------- 5
def fig_cusparselt():
    cs = load("cusparselt_stand.json")
    if not cs:
        return
    names = [r["name"] for r in cs]
    speedups = [r["speedup"] for r in cs]
    colors = [C_RECOVER if s > 1.0 else C_BAD for s in speedups]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, speedups, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for rect, s in zip(bars, speedups):
        ax.annotate(f"{s:.2f}×", xy=(s, rect.get_y() + rect.get_height()/2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=12)

    ax.axvline(1.0, ls="--", color=C_BASE, alpha=0.7, label="dense parity (1.0×)")
    ax.set_xlabel("speedup of cuSPARSELt sparse 2:4 vs dense fp16  (>1 = sparse wins)")
    ax.set_title("cuSPARSELt on RTX A4000 (sm_86): sparse 2:4 is slower than dense at every shape")
    ax.set_xlim(0, max(2.0, max(speedups) * 1.2))
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = OUT / "05_cusparselt_speedup.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------- 6
def fig_method_landscape():
    full = load("full_sparsity.json")
    ft = load("finetune.json")
    it = load("iterative.json")
    if not full:
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # full_sparsity: bs=8 only (one point per method)
    pts = []
    for k, v in full.items():
        method, bs = parse_full_key(k)
        if bs != 8:
            continue
        pts.append({"method": method,
                    "sparsity": v["sparsity"]["sparsity"],
                    "dice": v["bench"]["mean_dice"]})

    def color_for(m: str) -> str:
        if m.startswith("magnitude"): return C_MAG
        if m == "2to4":              return C_2TO4
        if m.startswith("structured"): return C_STRUCT
        return C_BASE

    for p in pts:
        ax.scatter(p["sparsity"], p["dice"], color=color_for(p["method"]),
                   s=140, alpha=0.85, edgecolors="black", linewidths=0.6,
                   label=p["method"] if p["method"] in {"baseline_fp16"} else None)

    # finetune post points (decoder-only sparsity)
    if ft:
        for r in ft:
            ax.scatter(r["sparsity_post"], r["post_dice"],
                       color=C_RECOVER, marker="*", s=250, edgecolors="black",
                       linewidths=0.7, alpha=0.95)
        ax.scatter([], [], color=C_RECOVER, marker="*", s=180, label="post fine-tune (decoder)")

    # iterative final point
    if it:
        last = it[-1]
        ax.scatter(last["sparsity"], last["post_dice"],
                   color=C_RECOVER, marker="P", s=250, edgecolors="black",
                   linewidths=0.7, label="iterative 70% (decoder)")

    # method legend dots
    for m, c, lbl in [("magnitude*", C_MAG,  "magnitude (no ft)"),
                      ("2to4",       C_2TO4, "2:4 (no ft)"),
                      ("structured", C_STRUCT, "structured (no ft)")]:
        ax.scatter([], [], color=c, s=140, edgecolors="black", linewidths=0.6, label=lbl)

    ax.axhline(0.9912, ls="--", color=C_BASE, alpha=0.6, label="baseline Dice 0.9912")
    ax.set_xlabel("sparsity (global, all conv weights)")
    ax.set_ylabel("Dice (BinaryF1)")
    ax.set_title("Method landscape — Dice vs sparsity, with and without fine-tune")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)
    fig.tight_layout()
    out = OUT / "06_method_landscape.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main() -> int:
    fig_quality_envelope()
    fig_finetune_recovery()
    fig_iterative_vs_oneshot()
    fig_asp_combo()
    fig_cusparselt()
    fig_method_landscape()
    return 0


if __name__ == "__main__":
    sys.exit(main())
