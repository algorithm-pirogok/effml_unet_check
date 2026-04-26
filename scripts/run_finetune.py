"""Sparse fine-tuning experiments.

For each sparsity method we:
  1. load `last.pth` weights into UNet 31M
  2. apply the sparsification *with the mask hook left active* (so optimizer
     steps cannot resurrect pruned weights)
  3. fine-tune for `EPOCHS` epochs on the Carvana train split
  4. evaluate Dice on the val split (same 400 images as `run_full.py`)

NVIDIA ASP recipe in spirit: prune once, fine-tune to recover, mask is
preserved by `torch.nn.utils.prune`'s PruningContainer (forward computes
`weight = weight_orig * weight_mask`, gradients flow through `weight_orig`,
and the zero entries of the mask are never updated effectively).

Output: results/finetune.json with per-method {pre_dice, post_dice, sparsity}.
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
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm

from src.carvana_dataset import CarvanaTrainDataset
from src.sparsity import _prunable_conv_modules, _apply_2of4_mask_to_tensor, measure_sparsity, CONV_TYPES
from src.unet import make_full_unet

WEIGHTS_PATH = Path(os.environ.get("UNET_WEIGHTS", str(ROOT / "weights" / "last.pth")))
CARVANA_TRAIN = Path(os.environ.get("CARVANA_TRAIN", str(ROOT / "data" / "carvana" / "train")))
CARVANA_MASKS = Path(os.environ.get("CARVANA_MASKS", str(ROOT / "data" / "carvana" / "train_masks")))

EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
BATCH = int(os.environ.get("FT_BS", "2"))   # contention is brutal; bs=2 to fit
LR = float(os.environ.get("FT_LR", "1e-4"))
SEED = 42


def build_loaders(scale: float = 0.5):
    full = CarvanaTrainDataset(
        train_root=str(CARVANA_TRAIN),
        mask_root=str(CARVANA_MASKS),
        scale=scale,
        max_samples=2000,
    )
    n_val = int(len(full) * 0.2)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    train, val = random_split(full, [n_train, n_val], generator=g)
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=4,
                            pin_memory=True)
    return train_loader, val_loader


def build_model(device: torch.device) -> nn.Module:
    m = make_full_unet(n_classes=2)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    m.load_state_dict(state, strict=True)
    return m.to(device)


def apply_magnitude_with_mask(model: nn.Module, amount: float) -> nn.Module:
    """Global L1 pruning, mask kept active (no `prune.remove` -> fine-tunable)."""
    prune.global_unstructured(
        _prunable_conv_modules(model),
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def apply_2to4_with_mask(model: nn.Module) -> nn.Module:
    """Apply a 2:4 mask via `prune.custom_from_mask`, fine-tunable."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CONV_TYPES):
                masked = _apply_2of4_mask_to_tensor(m.weight.data)
                mask = (masked != 0).to(m.weight.dtype)
                prune.custom_from_mask(m, name="weight", mask=mask)
    return model


def dice_loss(logits, targets, num_classes: int = 2, smooth: float = 1e-6):
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    num = 2 * (probs * targets_oh).sum(dims)
    den = probs.sum(dims) + targets_oh.sum(dims)
    return 1 - ((num + smooth) / (den + smooth)).mean()


@torch.no_grad()
def eval_dice(model, loader, device, num_classes: int = 2) -> float:
    model.eval()
    metric = BinaryF1Score().to(device)
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        preds = torch.argmax(logits, dim=1)
        metric.update((preds > 0).long().flatten(), (masks > 0).long().flatten())
    return float(metric.compute().item())


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc="train", leave=False)
    total_loss, seen = 0.0, 0
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = ce(logits, masks) + dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
        pbar.set_postfix(loss=loss.item())
    return total_loss / max(seen, 1)


def run_one(label: str, sparsify_fn, train_loader, val_loader, device) -> dict:
    print(f"\n=== fine-tune: {label} ===")
    model = build_model(device)
    sparsify_fn(model)
    sp = measure_sparsity(model)["global"]
    print(f"  sparsity: {sp['sparsity']:.4f}")

    pre = eval_dice(model, val_loader, device)
    print(f"  pre-finetune Dice : {pre:.4f}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    losses = []
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optim, scaler, device)
        d = eval_dice(model, val_loader, device)
        print(f"  epoch {ep+1}/{EPOCHS}: train_loss={loss:.4f}  val_dice={d:.4f}")
        losses.append({"epoch": ep + 1, "train_loss": loss, "val_dice": d})

    # check sparsity is preserved (it must be)
    sp_after = measure_sparsity(model)["global"]
    post = eval_dice(model, val_loader, device)
    print(f"  post-finetune Dice: {post:.4f}    sparsity_now: {sp_after['sparsity']:.4f}")

    del model
    torch.cuda.empty_cache()
    return {
        "label": label,
        "pre_dice": pre,
        "post_dice": post,
        "sparsity_pre": sp["sparsity"],
        "sparsity_post": sp_after["sparsity"],
        "epochs": losses,
    }


def main() -> int:
    device = torch.device("cuda")
    print(f"device: {device}  torch: {torch.__version__}  epochs: {EPOCHS}  bs: {BATCH}  lr: {LR}")

    train_loader, val_loader = build_loaders()
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    runs = []
    runs.append(run_one("magnitude@0.7",
                        lambda m: apply_magnitude_with_mask(m, 0.7),
                        train_loader, val_loader, device))
    runs.append(run_one("magnitude@0.9",
                        lambda m: apply_magnitude_with_mask(m, 0.9),
                        train_loader, val_loader, device))
    runs.append(run_one("2to4",
                        lambda m: apply_2to4_with_mask(m),
                        train_loader, val_loader, device))

    out_path = ROOT / "results" / "finetune.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(runs, indent=2))
    print(f"\n{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
