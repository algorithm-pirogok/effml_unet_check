"""Dense sparsity sweep, with and without fine-tuning.

Magnitude (L1) prune the FULL model at each amount in `AMOUNTS`. For each
amount we record:
  - pre_dice : Dice immediately after pruning (no retrain)
  - post_dice: Dice after 1 epoch decoder-only fine-tune (encoder frozen)

The two curves on a single sparsity axis show how fine-tuning shifts the
cliff to the right.

Output: results/sweep.json — one entry per amount.
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
    FrozenEncoderUNet,
    build_loaders,
    eval_dice,
    train_one_epoch,
)
from src.sparsity import _prunable_conv_modules, measure_sparsity
from src.unet import make_full_unet

WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
LR = float(os.environ.get("FT_LR", "2e-4"))
SEED = 42

AMOUNTS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97)


def fresh_full_model_pruned(device, amount: float):
    """Load fresh weights, prune the FULL model globally, wrap in FrozenEncoderUNet
    so fine-tune trains only the decoder (encoder weights stay zero+frozen)."""
    base = make_full_unet(n_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    base.load_state_dict(state, strict=True)
    base.to(device)

    prune.global_unstructured(
        _prunable_conv_modules(base),
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return FrozenEncoderUNet(base).to(device)


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  torch: {torch.__version__}  epochs/step: {EPOCHS}  lr: {LR}")
    train_loader, val_loader = build_loaders()
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    out: list[dict] = []

    # baseline (no prune)
    base_model = FrozenEncoderUNet(make_full_unet(n_classes=2)).to(device)
    base_state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    base_state.pop("mask_values", None)
    base_model.base.load_state_dict(base_state, strict=True)
    baseline_dice = eval_dice(base_model, val_loader, device)
    print(f"\nbaseline Dice (no prune): {baseline_dice:.4f}")
    out.append({"amount": 0.0, "sparsity": 0.0, "pre_dice": baseline_dice, "post_dice": baseline_dice})
    del base_model
    torch.cuda.empty_cache()

    for amount in AMOUNTS:
        print(f"\n=== amount={amount:.3f} ===")
        model = fresh_full_model_pruned(device, amount)
        sp = measure_sparsity(model)["global"]
        pre = eval_dice(model, val_loader, device)
        print(f"  sparsity: {sp['sparsity']:.4f}  pre Dice: {pre:.4f}")

        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda")
        for ep in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optim, scaler, device)
            print(f"  epoch {ep+1}: train_loss={loss:.4f}")

        post = eval_dice(model, val_loader, device)
        sp_after = measure_sparsity(model)["global"]
        print(f"  post Dice: {post:.4f}    sparsity: {sp_after['sparsity']:.4f}")

        out.append({
            "amount": amount,
            "sparsity": sp_after["sparsity"],
            "pre_dice": pre,
            "post_dice": post,
        })
        del model
        torch.cuda.empty_cache()

        # save partial each step in case of crash
        partial = ROOT / "results" / "sweep.partial.json"
        partial.parent.mkdir(exist_ok=True)
        partial.write_text(json.dumps(out, indent=2))

    out_path = ROOT / "results" / "sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{out_path}  ({len(out)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
