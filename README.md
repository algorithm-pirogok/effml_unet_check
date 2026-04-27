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

---

## Quantisation experiments (added)

Beyond the fp16 / `torch.compile` baseline, this repo now contains a full
INT8-quantisation study and a Triton-`tl.dot` diagnostic.

| Experiment | Code | Write-up |
|------------|------|----------|
| **TensorRT INT8** vs PyTorch-fp16 vs TRT-fp16 — full PTQ pipeline (ONNX export → `IInt8EntropyCalibrator2` → INT8/FP16 engine) on Colab T4. Decomposes the 4.5× speedup into ~2× from TRT fusion+NHWC and ~2× from INT8 IMMA. Δ Dice = **−0.0001**. | [`unet_segmentation_quantised.ipynb`](unet_segmentation_quantised.ipynb) | [`QUANTISATION_NOTES.md`](QUANTISATION_NOTES.md) |
| **Training from scratch** — CLI-mirror of the notebook's training cells (kept for reproducibility / potential future QAT, which PTQ made unnecessary). | [`training/`](training/) | [`training/README.md`](training/README.md) |
| **Triton int8 `tl.dot` on T4** — diagnostic that proves int8 dot is rejected at every tile by Triton 3.6.0's frontend (regression vs PR #2364). | [`trton_exp_of_dor_int8.py`](trton_exp_of_dor_int8.py) + [`trton_exp_of_dor_int8_colab_output.txt`](trton_exp_of_dor_int8_colab_output.txt) | [`TRITON_NOTES.md`](TRITON_NOTES.md) |

Bottom line: on Colab T4, **TensorRT INT8 PTQ gives 4.0–4.5× over PyTorch-fp16
without measurable quality loss**, and the path doesn't depend on Triton.
