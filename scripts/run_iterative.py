"""Iterative magnitude pruning vs one-shot.

Protocol (NVIDIA-style "gradual prune"):
  iter 1: prune to 30% sparsity (decoder), fine-tune 1 epoch
  iter 2: KEEP iter-1 weights, raise mask to 50% sparsity, fine-tune 1 epoch
  iter 3: KEEP iter-2 weights, raise mask to 70%, fine-tune 1 epoch

Comparison: final 70% iterative vs the one-shot 70% from run_finetune.py.
Same frozen-encoder wrapper to fit on a contended A4000.
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

# Reuse helpers from the one-shot runner.
sys.path.insert(0, str(ROOT / "scripts"))
from run_finetune import (
    FrozenEncoderUNet,
    apply_magnitude_with_mask,
    build_loaders,
    build_model,
    eval_dice,
    train_one_epoch,
)
from src.sparsity import _prunable_conv_modules, measure_sparsity

WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
EPOCHS_PER_STEP = int(os.environ.get("ITER_EPOCHS_PER_STEP", "1"))
BATCH = int(os.environ.get("FT_BS", "4"))
LR = float(os.environ.get("FT_LR", "2e-4"))
SEED = 42

LADDER = (0.3, 0.5, 0.7)


def freeze_current_mask(model: nn.Module) -> None:
    """Bake the active mask into weights, then drop the PruningContainer.

    After this, calling `prune.global_unstructured` again will produce a fresh
    mask whose bottom-amount selection naturally includes all current zeros
    (they have absolute value 0).
    """
    for m, name in _prunable_conv_modules(model):
        if hasattr(m, f"{name}_mask"):
            prune.remove(m, name)


def run() -> None:
    device = torch.device("cuda")
    print(f"device: {device}  torch: {torch.__version__}  bs: {BATCH}  lr: {LR}  epochs/step: {EPOCHS_PER_STEP}")

    train_loader, val_loader = build_loaders()
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    print("\n=== iterative ladder: 30% -> 50% -> 70% ===")
    model = build_model(device, frozen_encoder=True)

    history: list[dict] = []
    for step, target in enumerate(LADDER, 1):
        print(f"\n  step {step}: prune to {target:.0%} sparsity (decoder weights)")
        freeze_current_mask(model)
        apply_magnitude_with_mask(model, amount=target)
        sp = measure_sparsity(model)["global"]
        print(f"    sparsity now: {sp['sparsity']:.4f}")

        pre = eval_dice(model, val_loader, device)
        print(f"    pre-step Dice : {pre:.4f}")

        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda")
        train_loss = None
        for ep in range(EPOCHS_PER_STEP):
            train_loss = train_one_epoch(model, train_loader, optim, scaler, device)
            d = eval_dice(model, val_loader, device)
            print(f"    epoch {ep+1}: train_loss={train_loss:.4f}  val_dice={d:.4f}")

        post = eval_dice(model, val_loader, device)
        sp_after = measure_sparsity(model)["global"]
        print(f"    post-step Dice: {post:.4f}    sparsity: {sp_after['sparsity']:.4f}")
        history.append({
            "step": step,
            "target": target,
            "sparsity": sp_after["sparsity"],
            "pre_dice": pre,
            "post_dice": post,
            "train_loss": train_loss,
        })

    out = ROOT / "results" / "iterative.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(history, indent=2))
    print(f"\n{out}")


if __name__ == "__main__":
    run()
