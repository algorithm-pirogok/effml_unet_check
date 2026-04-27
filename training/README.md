# Training the Carvana UNet from scratch

CLI-version of cells 16 + 18 from
[`../unet_segmentation_quantised.ipynb`](../unet_segmentation_quantised.ipynb).
Same model, same loss, same optimiser, same AMP — just runnable outside Colab.

## Why this exists (and why it ended up unused)

Изначально мы держали в голове **Quantisation-Aware Training (QAT)** на
случай, если PTQ через TensorRT уронит качество. Поэтому отдельно
сохранили чистый обучающий пайплайн — чтобы можно было дообучить модель
с псевдо-квантизованными слоями (`torch.quantization.QAT` или подобное).

На практике **QAT не понадобился**: PTQ дал Δ Dice = −0.0001 (см. секцию
«UNet INT8 quantisation» в [`../README.md`](../README.md)). Файлы остаются
для воспроизводимости чекпоинта `last.pth` и на случай, если в будущем
модель/датасет станет более чувствительной к INT8.

## Files

| File         | Source in notebook |
|--------------|--------------------|
| `unet.py`    | cell 4   — `DoubleConv / Down / Up / OutConv / UNet`. |
| `dataset.py` | cell 10 + cell 12 — `CarvanaTrainDataset`, `set_seed`, `build_splits`. |
| `train.py`   | cells 16 + 18 — `dice_loss / train_one_epoch / evaluate` + main loop. |
| `validate.py`| cell 24 — `evaluate(..., amp_enabled=False)` for fp32 baseline Dice. |

## Usage

Defaults match the notebook (3 epochs, batch=2, AdamW lr=1e-4, fp16 AMP,
`val_percent=0.2`, seed=42 — so the train/val split is identical):

```bash
python train.py \
    --train-root /content/drive/MyDrive/carvana/train \
    --mask-root  /content/drive/MyDrive/carvana/train_masks \
    --save-dir   ./checkpoints

python validate.py \
    --train-root /content/drive/MyDrive/carvana/train \
    --mask-root  /content/drive/MyDrive/carvana/train_masks \
    --checkpoint ./checkpoints/last.pth
```

## Saved artifacts

`train.py` writes three files into `--save-dir` each epoch (same names as
the notebook):

| File | Contents |
|------|----------|
| `unet_carvana_self_trained.pth` | best (by val Dice) plain `state_dict` |
| `last.pth`                      | `state_dict` after the last epoch |
| `training_state.pth`            | full state (model, optim, scheduler, scaler, history) |

The TensorRT-INT8 cell in the notebook reads `last.pth` to build the
fp32 ONNX. Train this script first if you don't already have a checkpoint.
