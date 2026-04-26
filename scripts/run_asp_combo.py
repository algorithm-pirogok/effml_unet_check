"""NVIDIA ASP-style combo: magnitude → fine-tune → 2:4 → fine-tune.

The full ASP recipe in spirit:
  1. magnitude-prune the decoder to 50% (drop the lowest L1 weights globally)
  2. short fine-tune to recover Dice
  3. apply 2:4 mask *on top of the magnitude-pruned weights*
     (after `prune.remove`, the magnitude zeros are baked in; the 2:4 selector
     naturally keeps them — they are the lowest |x| in their 4-blocks)
  4. another fine-tune to recover after the 2:4 step

Final: 50% magnitude AND 2:4 pattern → at least 50% sparsity (often more,
because some magnitude zeros land inside 2:4 groups too).

Same frozen-encoder wrapper as run_finetune / run_iterative to fit memory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, str(ROOT / "scripts"))
from run_finetune import (
    apply_2to4_with_mask,
    apply_magnitude_with_mask,
    build_loaders,
    build_model,
    eval_dice,
    train_one_epoch,
)
from src.sparsity import _prunable_conv_modules, measure_sparsity

EPOCHS_PER_STEP = int(os.environ.get("ASP_EPOCHS_PER_STEP", "1"))
BATCH = int(os.environ.get("FT_BS", "4"))
LR = float(os.environ.get("FT_LR", "2e-4"))
MAGNITUDE_TARGET = float(os.environ.get("ASP_MAGNITUDE", "0.5"))


def freeze_current_mask(model: nn.Module) -> None:
    for m, name in _prunable_conv_modules(model):
        if hasattr(m, f"{name}_mask"):
            prune.remove(m, name)


def fine_tune(model, train_loader, val_loader, device, epochs: int) -> dict:
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, optim, scaler, device)
        d = eval_dice(model, val_loader, device)
        history.append({"epoch": ep + 1, "train_loss": loss, "val_dice": d})
        print(f"      epoch {ep+1}: train_loss={loss:.4f}  val_dice={d:.4f}")
    return {"history": history, "final_dice": history[-1]["val_dice"] if history else None}


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  torch: {torch.__version__}  bs: {BATCH}  lr: {LR}  "
          f"epochs/step: {EPOCHS_PER_STEP}  magnitude target: {MAGNITUDE_TARGET}")

    train_loader, val_loader = build_loaders()
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    print("\n=== ASP combo: magnitude + 2:4 + fine-tune ===")
    model = build_model(device, frozen_encoder=True)

    out = {"steps": []}

    # baseline (no sparsity, just decoder-only)
    pre0 = eval_dice(model, val_loader, device)
    print(f"  step 0 (no prune)  Dice: {pre0:.4f}")
    out["baseline_decoder"] = pre0

    # step 1: magnitude
    print(f"\n  step 1: magnitude prune decoder to {MAGNITUDE_TARGET:.0%}")
    apply_magnitude_with_mask(model, amount=MAGNITUDE_TARGET)
    sp1 = measure_sparsity(model)["global"]
    pre1 = eval_dice(model, val_loader, device)
    print(f"    sparsity: {sp1['sparsity']:.4f}  pre-tune Dice: {pre1:.4f}")
    ft1 = fine_tune(model, train_loader, val_loader, device, EPOCHS_PER_STEP)
    sp1_after = measure_sparsity(model)["global"]
    out["steps"].append({
        "name": "magnitude+ft",
        "sparsity": sp1_after["sparsity"],
        "pre_tune_dice": pre1,
        "post_tune_dice": ft1["final_dice"],
        "history": ft1["history"],
    })

    # step 2: bake magnitude mask, then add 2:4 on top
    print(f"\n  step 2: bake magnitude zeros, then apply 2:4 mask on top")
    freeze_current_mask(model)
    apply_2to4_with_mask(model)
    sp2 = measure_sparsity(model)["global"]
    pre2 = eval_dice(model, val_loader, device)
    print(f"    sparsity: {sp2['sparsity']:.4f}  pre-tune Dice: {pre2:.4f}")
    ft2 = fine_tune(model, train_loader, val_loader, device, EPOCHS_PER_STEP)
    sp2_after = measure_sparsity(model)["global"]
    out["steps"].append({
        "name": "magnitude+2to4+ft",
        "sparsity": sp2_after["sparsity"],
        "pre_tune_dice": pre2,
        "post_tune_dice": ft2["final_dice"],
        "history": ft2["history"],
    })

    out_path = ROOT / "results" / "asp_combo.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
