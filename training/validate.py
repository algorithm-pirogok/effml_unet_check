"""Validation script — mirrors `unet_segmentation_quantised.ipynb` cell 24.

Same `evaluate()` as in the training script (which is verbatim from cell 16),
called with `amp_enabled=False` to produce the fp32 baseline Dice that all
fp16 / TRT-fp16 / TRT-int8 numbers in the notebook are compared against.

Same seeded train/val split as `train.py` — with matching --seed and
--val-percent the evaluated images are identical to the held-out 20% used
during training.

Example:
  python validate.py \\
    --train-root /content/drive/MyDrive/carvana/train \\
    --mask-root  /content/drive/MyDrive/carvana/train_masks \\
    --checkpoint ./checkpoints/last.pth
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import CarvanaTrainDataset, build_splits, set_seed, SEED
from train import evaluate
from unet import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", required=True)
    p.add_argument("--mask-root",  required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--val-percent", type=float, default=0.2)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--amp", action="store_true",
                   help="Run with fp16 AMP (default off — matches the notebook's pre-benchmark fp32 sanity).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device={device}  torch={torch.__version__}  amp_enabled={args.amp}")

    full_dataset = CarvanaTrainDataset(
        train_root=args.train_root, mask_root=args.mask_root, scale=args.scale,
    )
    _, val_dataset = build_splits(full_dataset, args.val_percent, args.seed)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    model = UNet(n_channels=3, n_classes=args.n_classes).to(
        device, memory_format=torch.channels_last
    )
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)

    val_loss, val_dice = evaluate(
        model, val_loader, device, amp_enabled=args.amp, num_classes=args.n_classes,
    )
    print(f"val | loss={val_loss:.4f} | Dice={val_dice:.4f}")


if __name__ == "__main__":
    main()
