"""Train UNet on Carvana — CLI script that mirrors `unet_segmentation_quantised.ipynb`
cells 16 + 18 exactly (same loss, optimiser, scheduler, AMP, channels_last,
checkpoint layout). The notebook trains in-line with `clear_output` + matplotlib
re-rendering; this file replaces that with `print()` so it runs in any shell.

Defaults exactly match the notebook:
  - 3 epochs, batch_size=2, AdamW(lr=1e-4, wd=1e-4)
  - ReduceLROnPlateau on val Dice (mode=max, patience=2, factor=0.5)
  - CrossEntropyLoss + soft-Dice loss
  - fp16 AMP on CUDA, channels_last memory format, grad_clip=1.0
  - val_percent=0.2, seed=42 (so train/val split matches the notebook)

Saves three files into --save-dir each epoch:
  unet_carvana_self_trained.pth — best (by val Dice) state_dict
  last.pth                      — state_dict after the last completed epoch
  training_state.pth            — full state for clean resume

Example:
  python train.py \\
    --train-root /content/drive/MyDrive/carvana/train \\
    --mask-root  /content/drive/MyDrive/carvana/train_masks \\
    --save-dir   ./checkpoints
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm

from dataset import CarvanaTrainDataset, build_splits, set_seed, SEED
from unet import UNet


# ---------------------------------------------------------------------------
# Loss / loop helpers — verbatim from notebook cell 16.
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """Soft multiclass Dice loss: 1 - mean dice over classes + batch."""
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    num = 2 * (probs * targets_oh).sum(dims)
    den = probs.sum(dims) + targets_oh.sum(dims)
    dice = (num + smooth) / (den + smooth)
    return 1 - dice.mean()


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, grad_clip=1.0):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, seen = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = ce(logits, masks) + dice_loss(logits, masks, num_classes=model.n_classes)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
        pbar.set_postfix(loss=loss.item())
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, num_classes=2):
    """Return (avg_loss, foreground_dice) on the given loader."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    metric = BinaryF1Score().to(device)
    total_loss, seen = 0.0, 0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = ce(logits, masks) + dice_loss(logits, masks, num_classes=num_classes)
        preds = torch.argmax(logits, dim=1)
        metric.update((preds > 0).long().flatten(), (masks > 0).long().flatten())
        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
    return total_loss / max(seen, 1), float(metric.compute().item())


# ---------------------------------------------------------------------------
# CLI driver — assembles the same training run as cell 18.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", required=True, help="Carvana train images dir")
    p.add_argument("--mask-root",  required=True, help="Carvana train masks dir")
    p.add_argument("--save-dir",   default="./checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--val-percent", type=float, default=0.2)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    amp_enabled = (device.type == "cuda")
    print(f"device={device}  amp={amp_enabled}  torch={torch.__version__}")

    full_dataset = CarvanaTrainDataset(
        train_root=args.train_root,
        mask_root=args.mask_root,
        scale=args.scale,
    )
    train_dataset, val_dataset = build_splits(full_dataset, args.val_percent, args.seed)
    print(f"full: {len(full_dataset)}  train: {len(train_dataset)}  val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=amp_enabled, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=amp_enabled,
        persistent_workers=True,
    )

    model = UNet(n_channels=3, n_classes=args.n_classes).to(
        device, memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path     = os.path.join(args.save_dir, "unet_carvana_self_trained.pth")
    last_path           = os.path.join(args.save_dir, "last.pth")
    training_state_path = os.path.join(args.save_dir, "training_state.pth")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled, args.grad_clip
        )
        val_loss, val_dice = evaluate(
            model, val_loader, device, amp_enabled, num_classes=args.n_classes
        )
        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)

        torch.save(model.state_dict(), last_path)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "history": history,
        }, training_state_path)

        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_dice={val_dice:.4f}  (best={best_dice:.4f})")

    print(f"\nBest val Dice: {best_dice:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)


if __name__ == "__main__":
    main()
