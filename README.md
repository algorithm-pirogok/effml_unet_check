# UNet Segmentation Inference Benchmarks

Benchmarks the pretrained Carvana UNet model from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) on a segmentation task.

Compares:
- **Baseline fp16 inference** (when CUDA is available)
- **`torch.compile`-accelerated inference** (when supported by the installed PyTorch)

## Project structure

```
src/
  model.py        — pretrained UNet loading
  dataset.py      — CarvanaTrainDataset and DataLoader builder
  benchmark.py    — inference benchmark runner with Dice/F1 metric
  experiment.py   — experiment orchestration (configs, history)
run_benchmark.py  — main entry point: runs all experiments and saves results
plot_results.py   — plots avg time per image vs batch size
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_benchmark.py
python plot_results.py
```

The benchmark script expects Carvana dataset images and masks. Update `train_root` and `mask_root` paths in `run_benchmark.py` accordingly.
