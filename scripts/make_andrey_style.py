"""Generate Andrey-style PNG slides for the sparsity block.

Layout matches `Unet inference.pdf`:
  - 16:9, plain white background, sans-serif
  - title top-left
  - bullets with ● markers; green/red emphasis spans
  - matplotlib panels embedded as PIL images
  - one-line caption below

Output: results/slides/andrey_style/*.png — drop each PNG into Google Slides
as a full-bleed image.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "slides" / "andrey_style"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Match Andrey's deck: 16:9, white bg, sans-serif similar to Arial.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

W, H = 10, 5.625                  # match Andrey's 720x405 pts page exactly
DPI = 200                         # 2000x1125 px — sharp on Google Slides
TITLE_X, TITLE_Y = 0.05, 0.9
TITLE_FS = 32
BODY_FS = 22
CAPTION_FS = 20

GREEN  = "#2c8a3e"
RED    = "#c0392b"
GRAY   = "#666666"


def new_slide():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def title(ax, text):
    ax.text(TITLE_X, TITLE_Y, text, fontsize=TITLE_FS, weight="normal",
            color="black", va="bottom", ha="left")


def bullets(ax, items, x=0.07, y_start=0.78, line_h=0.075):
    """items: list of (text, color or None)."""
    for i, (txt, color) in enumerate(items):
        y = y_start - i * line_h
        ax.text(x, y, "●", fontsize=BODY_FS, color=color or "black", va="top")
        ax.text(x + 0.025, y, txt, fontsize=BODY_FS,
                color=color or "black", va="top")


def caption(ax, text, x=TITLE_X, y=0.05):
    ax.text(x, y, text, fontsize=CAPTION_FS, color="black", va="bottom")


def embed_image(ax, path: Path, x, y, w, h):
    """Place a PNG inside the slide at fractional coords."""
    img = mpimg.imread(path)
    sub = ax.figure.add_axes([x, y, w, h])
    sub.imshow(img)
    sub.axis("off")


def save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out, dpi=DPI, bbox_inches=None, pad_inches=0,
                facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


# A_section_title removed: Andrey's deck flows directly into the next topic
# without divider slides, so a section title would break the visual rhythm.


# ------------------------------------------------------------------ B. Methods
def slide_b_methods():
    fig, ax = new_slide()
    title(ax, "Использованные методы спарсификации")
    items = [
        ("Magnitude pruning (global L1 unstructured, torch.nn.utils.prune)", GREEN),
        ("2:4 semi-structured (NVIDIA Ampere pattern, 50% sparsity)", GREEN),
        ("Structured channel pruning (per-conv L2 norm, mask-only)", GREEN),
        ("Fine-tune с frozen encoder (NVIDIA ASP recipe, 1 эпоха)", GREEN),
        ("Iterative ladder и ASP combo (поверх базовых методов)", GREEN),
    ]
    bullets(ax, items, y_start=0.76, line_h=0.085)
    caption(ax, "Set-up: UNet 31M, Carvana val=400, NVIDIA RTX A4000 (sm_86), fp16, bs=8")
    save(fig, "B_methods.png")


# ------------------------------------------------------------------ C. Quality envelope
def slide_c_envelope():
    fig, ax = new_slide()
    title(ax, "Magnitude pruning: качество vs sparsity")
    img_path = RES / "slides" / "01_quality_envelope.png"
    embed_image(ax, img_path, x=0.10, y=0.13, w=0.80, h=0.70)
    caption(ax, "Без retrain cliff на 75–80%, после 1 эпохи fine-tune Dice держит baseline до 97%")
    save(fig, "C_quality_envelope.png")


# ------------------------------------------------------------------ D. Iterative + ASP
def slide_d_iter_asp():
    fig, ax = new_slide()
    title(ax, "Iterative ladder и NVIDIA ASP combo")
    embed_image(ax, RES / "slides" / "03_iterative_vs_oneshot.png",
                x=0.04, y=0.18, w=0.46, h=0.58)
    embed_image(ax, RES / "slides" / "04_asp_combo.png",
                x=0.52, y=0.18, w=0.46, h=0.58)
    caption(ax, "Iterative 30→50→70% даёт +0.001 к one-shot. ASP combo: 0.991 → 0.994 при 24% sparsity")
    save(fig, "D_iterative_asp.png")


# ------------------------------------------------------------------ E. cuSPARSELt
def slide_e_cusparselt():
    fig, ax = new_slide()
    title(ax, "cuSPARSELt 2:4 — hardware path на A4000")
    embed_image(ax, RES / "slides" / "05_cusparselt_speedup.png",
                x=0.10, y=0.16, w=0.80, h=0.68)
    caption(ax, "На consumer Ampere (sm_86) sparse 2:4 везде медленнее dense, hardware path только на A100 (sm_80)")
    save(fig, "E_cusparselt.png")


# ------------------------------------------------------------------ F. Profiler + memory table
def slide_f_profiler():
    fig, ax = new_slide()
    title(ax, "PyTorch Profiler и память")

    # left: profiler table
    rows = [
        ("baseline_fp16", "1.085 s", "~95%"),
        ("magnitude@0.5", "1.078 s", "~95%"),
        ("2:4 (50%)",     "1.076 s", "~95%"),
    ]
    head_y = 0.72
    ax.text(0.07, head_y, "Self CUDA total, 4 батча bs=8:",
            fontsize=BODY_FS, color="black", va="top", weight="bold")
    ax.text(0.07, head_y - 0.07, "конфиг",      fontsize=BODY_FS, color=GRAY)
    ax.text(0.27, head_y - 0.07, "CUDA total",  fontsize=BODY_FS, color=GRAY)
    ax.text(0.42, head_y - 0.07, "aten::conv2d", fontsize=BODY_FS, color=GRAY)
    for i, (name, total, share) in enumerate(rows):
        y = head_y - 0.16 - i * 0.07
        ax.text(0.07, y, name,  fontsize=BODY_FS)
        ax.text(0.27, y, total, fontsize=BODY_FS)
        ax.text(0.42, y, share, fontsize=BODY_FS)

    # right: bullet conclusions
    concl_x = 0.58
    items = [
        ("Разница <1% между всеми тремя", RED),
        ("~95% времени в aten::conv2d", None),
        ("cudnn dense kernels игнорируют нули", None),
        ("Для speedup нужны sparse kernels", None),
        ("Peak GPU memory: 6112 MB у всех", RED),
    ]
    for i, (txt, color) in enumerate(items):
        y = 0.66 - i * 0.075
        ax.text(concl_x, y, "●", fontsize=BODY_FS, color=color or "black", va="top")
        ax.text(concl_x + 0.025, y, txt, fontsize=BODY_FS,
                color=color or "black", va="top")

    caption(ax, "Без sparse-aware kernels нет ни ускорения, ни экономии памяти")
    save(fig, "F_profiler.png")


# ------------------------------------------------------------------ G. Final summary
def slide_g_summary():
    fig, ax = new_slide()
    title(ax, "Итоги спарсификации")
    items = [
        ("Качество: 70% magnitude бесплатно, 97% после retrain ≥ baseline", GREEN),
        ("Скорость на dense fp16: ускорения нет", RED),
        ("cuSPARSELt 2:4 на A4000: 1.5–7× медленнее dense", RED),
        ("Peak GPU memory: одинаковый у всех методов (6112 MB)", RED),
        ("Сэкономили только размер чекпоинта (если хранить как sparse)", None),
        ("Дальше: physical channel removal, перенос на A100", None),
    ]
    bullets(ax, items, y_start=0.76, line_h=0.085)
    caption(ax, "Спарсификация без специальных kernels уменьшает только размер чекпоинта, не runtime")
    save(fig, "G_summary.png")


def main() -> int:
    slide_b_methods()
    slide_c_envelope()
    slide_d_iter_asp()
    slide_e_cusparselt()
    slide_f_profiler()
    slide_g_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
