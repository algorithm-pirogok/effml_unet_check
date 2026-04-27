"""Add extra sparsity points to an existing sweep.json.

Reads results/sweep.json, runs only the EXTRA amounts that aren't there yet,
appends them, and re-saves.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
sys.path.insert(0, str(ROOT / "scripts"))
from run_finetune import build_loaders, eval_dice, train_one_epoch
from run_sweep import fresh_full_model_pruned
from src.sparsity import measure_sparsity

EXTRA = (0.775, 0.99)
EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
LR = float(os.environ.get("FT_LR", "2e-4"))


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  extra amounts: {EXTRA}")
    train_loader, val_loader = build_loaders()

    sweep_path = ROOT / "results" / "sweep.json"
    out = json.loads(sweep_path.read_text()) if sweep_path.exists() else []
    existing = {round(r["amount"], 4) for r in out}

    for amount in EXTRA:
        if round(amount, 4) in existing:
            print(f"  skipping {amount} (already in sweep.json)")
            continue
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

    out.sort(key=lambda r: r["sparsity"])
    sweep_path.write_text(json.dumps(out, indent=2))
    print(f"\n{sweep_path}  ({len(out)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
