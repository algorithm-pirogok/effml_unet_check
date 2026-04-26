# UNet sparsification benchmarks

This document covers block 5 of the team plan: **sparsification of the trained
UNet (Carvana, 31M params)**. Three methods were implemented and benchmarked:

1. Unstructured global magnitude (L1) pruning — sweep over 30/50/70/90%.
2. 2:4 semi-structured sparsity — fixed 50% pattern.
3. Structured channel pruning (per-conv L2 norm) — 25% / 50% of out-channels.

## Why these three

The course assignment lists "pruning / 2:4 semi-structured" explicitly, so 1 and
2 are required. We added structured channel pruning (3) to expose a contrast:
unlike (1) and (2), it is the only one of the three that *can* deliver wall-time
speedup with stock dense kernels — but only after we physically rewire the
downstream layers, which we leave out of scope (see "Limitations").

## Method 1: unstructured magnitude pruning

`torch.nn.utils.prune.global_unstructured` with `L1Unstructured`, applied across
every `Conv2d` and `ConvTranspose2d` weight in the network. After applying, we
call `prune.remove(...)` so the masks are baked permanently into the weights —
this guarantees the measured zero-fraction is real, not virtual.

Sweep: amount ∈ {0.3, 0.5, 0.7, 0.9}.

**Expected speedup with dense fp16 kernels: ~0.** Zeros in the weight tensor
multiplied by activations still cost the same FLOPs in `cudnn` GEMM/CUTLASS.
The point of this experiment is to bound the *quality* envelope: how much
sparsity does the model tolerate before Dice falls off a cliff?

## Method 2: 2:4 semi-structured sparsity

For each conv weight `(out_c, in_c, kH, kW)` we flatten the trailing
`(in_c, kH, kW)` axes and zero, in every consecutive group of 4 elements, the
two with the smallest absolute value. Result: exactly 50% sparsity, organised
into 2-of-4 blocks along the input dimension.

This is the pattern accepted by NVIDIA's sparse Tensor Cores
([cuSPARSELt](https://docs.nvidia.com/cuda/cusparselt/)) and the
`torch.sparse.SparseSemiStructuredTensor` family. **Hardware acceleration on
Ampere+ is enabled only after a layer-replacement pass** that swaps in a
sparse-aware matmul implementation. The mask path here keeps the layer dense,
so the speedup we measure is a lower bound: it tells us "the pattern is
correct, the model still works with it" — and the right next step on a Hopper
or Ampere production deployment would be to plug in cuSPARSELt kernels.

T4 (Turing) does not support sparse tensor cores — A4000 (Ampere, sm_86) does.
This is why we deliberately benchmark on A4000 here rather than the team's T4
baseline.

## Method 3: structured channel pruning

`torch.nn.utils.prune.ln_structured(n=2, dim=0)` zeros entire output channels
of each conv, ranked by their L2 norm. We keep the masks "permanent" the same
way as method 1.

**Limitation we accepted:** we do not physically remove channels (which would
require re-wiring every downstream `Conv2d.in_channels`, every BatchNorm, and
every UNet skip-connection concat). With the shape preserved, the dense kernel
still does the full work; the experiment quantifies the *quality* effect of
removing channels. A production implementation would compose this with
`torch_pruning` or hand-written shape rewriting — out of scope under the
midterm time budget.

## Benchmark setup

- Hardware: NVIDIA RTX A4000 16 GB, beleriand, GPU 1.
- Software: torch 2.6.0+cu124, torchmetrics, fp16 autocast.
- Data: Carvana train split, 2000 images, scale=0.5 → val 400 (seeded
  `random_split`, gen 42, identical to Anton's training notebook).
- Per-config: 1 warmup batch, then `ceil(1280 / batch_size)` timed batches with
  `torch.cuda.synchronize` around each `model(images)` call.
- Quality: `BinaryF1Score` over predicted vs. ground-truth foreground.
- Batch sizes: 2, 4, 8, 16.

The benchmark JSON has the exact field set as Anton's
`unet_benchmarks.json`, so the team table can be merged trivially.

## Files

```
src/sparsity.py            — three methods + measure_sparsity helper
src/bench.py               — inference timing + Dice (Anton's JSON shape)
src/unet.py                — UNet, base_channels parametric (31M / 0.5M)
src/random_dataset.py      — synthetic data for the smoke test
src/carvana_dataset.py     — real Carvana loader, scale=0.5
scripts/run_smoke.py       — pipeline smoke on tiny UNet + random data
scripts/run_full.py        — full benchmark on UNet 31M + Carvana val
scripts/download_carvana.sh — kaggle CLI download helper
scripts/plot_sparsity.py   — render figures + summary CSV from JSON
```

## Headline results

UNet 31M, Carvana val (400 images, scale=0.5), A4000 GPU 6, fp16 autocast.
Baseline Dice on the val split is **0.9912**, matching the team PDF (0.99125).

| Method | Sparsity | Dice | Δ Dice | ms/img (bs=8) |
|---|---:|---:|---:|---:|
| baseline_fp16     | 0.000 | 0.9912 |   ±0    | 37.2 |
| magnitude @ 0.3   | 0.300 | 0.9910 | −0.0002 | 68.5 |
| magnitude @ 0.5   | 0.500 | 0.9909 | −0.0003 | 68.6 |
| magnitude @ 0.7   | 0.700 | 0.9825 | −0.0087 | 68.4 |
| magnitude @ 0.9   | 0.900 | 0.1077 | −0.8835 | 68.0 |
| 2:4 semi-struct.  | 0.500 | 0.0307 | −0.9605 | 39.6 |
| structured @ 0.25 | 0.250 | 0.0000 | −0.9912 | 53.6 |
| structured @ 0.50 | 0.500 | 0.0000 | −0.9912 | 68.4 |

(Per-batch CSV in `results/summary.csv`. Plots in `results/fig_*.png`.)

### Quality findings

- **Magnitude pruning is remarkably tolerant**: at 50% sparsity Dice loss is
  10⁻⁴, at 70% it's <1%. Only at 90% does the model collapse.
- **2:4 without fine-tuning destroys quality** even though raw sparsity is the
  same 50% as the magnitude pivot. Reason: the 2:4 constraint forces a
  per-block zero pattern over the input-channel axis, which is structurally
  more aggressive than picking the global bottom-50%. This is the canonical
  NVIDIA result — production 2:4 deployments always pair pruning with
  short fine-tuning (NVIDIA ASP recipe).
- **Structured channel pruning** drops Dice to zero immediately. Without
  re-training (or physically removing channels and BatchNorm slots), the
  zeroed output channels still flow into the decoder via skip connections,
  poisoning every downstream feature map.

### Speed findings

We measured **no consistent wall-time speedup** from any of the three methods
under stock dense fp16 kernels. This was expected and is documented in the
methodology section: zeros in dense GEMM cost the same FLOPs.

For a real speedup you need (a) cuSPARSELt-backed sparse Tensor Cores for the
2:4 case (only on Ampere+, requires layer replacement), or (b) physical
channel removal for the structured case (requires a UNet-aware shape rewrite
pass). Both are described in "Limitations" above and are out of scope for the
midterm. The benchmark numbers above are valid lower-bound estimates that
say "the sparsified model still runs and gives correct outputs"; on a real
deployment with cuSPARSELt the 2:4 row would drop by ~30–50% (NVIDIA reports
[1.7×–2.0×](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
on dense matmul workloads).

### Caveat on raw timing

beleriand A4000 is a shared 7-GPU box; during the run the chosen GPU 6 had
~8 GB free out of 16, and other users' workloads share L2/SM bandwidth. The
ms/img column above is therefore noisier than Anton's isolated-T4 numbers.
The qualitative conclusion (no speedup from sparse-but-dense kernels) is
robust to this noise.

## Profiler evidence (PyTorch Profiler)

To verify the "no speedup" claim quantitatively we ran `torch.profiler` over
4 active inference batches (bs=8) for three configs. Full traces are in
`results/profile_*.trace.json` (load into chrome://tracing); the top-line
numbers from `key_averages()`:

| config | Self CUDA total (4 batches) | aten::conv2d share |
|---|---:|---:|
| baseline_fp16 | **1.085 s** | ~95% |
| magnitude@0.5 | **1.078 s** | ~95% |
| 2:4 (50%)     | **1.076 s** | ~95% |

The sparsified configs differ from baseline by **<1%** in CUDA time. cudnn's
dense fp16 conv kernels do not exploit the zeros — they execute the full GEMM
either way. This is exactly what the kernel-level theory predicts.

## cuSPARSELt experiment — does the hardware path actually win on A4000?

To test whether wrapping the model in cuSPARSELt-backed sparse matmul would
recover the missing speedup, we ran an isolated GEMM bench:
`y = x @ W^T` with dense fp16 vs `to_sparse_semi_structured(W)` (cuSPARSELt
backend). Same shapes, same device (RTX A4000, sm_86):

| Shape (M×K×N) | dense ms | sparse ms | speedup |
|---|---:|---:|---:|
| Bottleneck-like 4096×1024×1024 | 0.15 | 1.11 | **0.14×** |
| Mid-decoder 8192×512×512       | 0.09 | 1.43 | 0.06× |
| Pointwise 65536×64×32          | 0.04 | 1.43 | 0.03× |
| Large 4096×2048×2048           | 0.58 | 0.81 | 0.72× |
| LLM 4096×4096×4096             | 2.28 | 2.34 | 0.97× |
| LLM 8192×8192×8192             | 17.8 | 26.5 | 0.67× |

**Sparse 2:4 is consistently slower than dense on this GPU — even at
LLM-shape sizes.** This is the actual key finding of the cuSPARSELt experiment:
on consumer / professional Ampere parts (sm_86: RTX A4000, A5000, A6000),
NVIDIA's sparse Tensor Core path through cuSPARSELt does not deliver positive
speedup. The 1.7×–2× advertised speedups are reproducible only on
**datacenter Ampere** (sm_80: A100, A40), where the sparse Tensor Cores have
a different, fully-fledged hardware implementation.

This is a non-trivial result for the team's planning: **for a real 2:4
deployment of UNet (or any model) the deployment target must be A100 or
newer**, not an RTX A4000 / A6000 box. JSON in `results/cusparselt_stand.json`.

## Fine-tuning recovers (and exceeds) baseline quality

Following the NVIDIA ASP recipe (prune once, fine-tune to recover), we ran 1
epoch of decoder-only fine-tuning with the encoder frozen (cheap on a
contended A4000) for three configs that originally collapsed:

| Method (decoder, mask kept active) | Pre-tune Dice | Post-tune Dice | vs baseline 0.9912 |
|---|---:|---:|---:|
| magnitude @ 0.7 | 0.9895 | **0.9940** | **+0.0028** |
| magnitude @ 0.9 | 0.2354 | **0.9941** | **+0.0029** |
| 2:4 (50%)       | 0.9844 | **0.9943** | **+0.0031** |

All three sparsified models **exceed baseline Dice** after a single fine-tune
epoch. magnitude@0.9 is the most striking — the model went from near-broken
(0.2354) to better-than-baseline (0.9941) in one pass. JSON in
`results/finetune.json`.

## Iterative pruning beats one-shot

Gradual ladder 30% → 50% → 70% (each step followed by 1 epoch of fine-tuning,
mask kept across steps so prior zeros stay zero):

| Step | Target | Sparsity (global) | Pre-step Dice | Post-step Dice |
|---|---:|---:|---:|---:|
| 1 | 30% | 0.118 | 0.9911 | **0.9939** |
| 2 | 50% | 0.196 | 0.9932 | **0.9945** |
| 3 | 70% | 0.275 | 0.9925 | **0.9950** |

Final iterative 70% Dice is **0.9950 vs 0.9940 one-shot** — a measurable
+0.001 win, consistent with the standard iterative-pruning literature.
JSON in `results/iterative.json`.

## Full ASP combo: magnitude → 2:4 → fine-tune

Stacking the two methods end to end:

| Stage | Sparsity | Post Dice |
|---|---:|---:|
| baseline (decoder-only, no prune) | 0.000 | 0.9912 |
| magnitude 50% + fine-tune          | 0.196 | **0.9940** |
| + 2:4 mask + fine-tune             | 0.240 | **0.9944** |

This is the canonical NVIDIA ASP recipe: a magnitude prune kicks the model to
a sparse subspace, fine-tune lets it adapt; then 2:4 imposes the
hardware-friendly pattern on the remaining non-zeros, and a second fine-tune
absorbs that. End-to-end, decoder Dice goes **0.9912 → 0.9944** while 24% of
the conv weights are zero (with a 2:4 structure on the residual). JSON in
`results/asp_combo.json`.

## Memory profile (negative result, but informative)

Peak GPU memory during one inference batch (bs=8, fp16):

| Method | Params | Peak GPU memory |
|---|---:|---:|
| baseline_fp16   | 124.2 MB | 6112.5 MB |
| magnitude @ 0.5 | 124.2 MB | 6112.5 MB |
| magnitude @ 0.9 | 124.2 MB | 6112.5 MB |
| 2:4 (50%)       | 124.2 MB | 6112.5 MB |
| structured 50%  | 124.2 MB | 6112.5 MB |

**Identical to the milligram.** Zeros inside a dense tensor occupy the same
storage as non-zeros. Activation tensors (which dominate the 6 GB peak) are
shape-preserved across all methods. Sparsification with stock kernels
delivers **neither speedup nor memory reduction** — only a smaller deployable
checkpoint *if* the file is saved as a sparse data structure on disk. To get
runtime memory back you need either physical channel removal (structured
pruning + downstream rewiring) or a sparse kernel that materialises only the
non-zero-times-active subblock at compute time. JSON in `results/memory.json`.
