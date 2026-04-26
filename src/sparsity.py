"""Sparsification methods for the UNet inference benchmark.

Three families:
  1. Unstructured global magnitude (L1) pruning — torch.nn.utils.prune.
  2. 2:4 semi-structured sparsity — column-wise 2-of-4 zero pattern over
     in-channel dim of conv weights. Mask applied permanently. On Ampere+ with
     proper layer replacement this enables sparse Tensor Core acceleration; on
     dense kernels the speedup is ~0 but the pattern is correct.
  3. Structured channel pruning — drop the lowest-Ln-norm output channels of
     each Conv2d (mask only, no physical removal — speedup requires re-wiring).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


CONV_TYPES = (nn.Conv2d, nn.ConvTranspose2d)


def _prunable_conv_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Return [(module, "weight"), ...] for every Conv2d / ConvTranspose2d."""
    out: List[Tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, CONV_TYPES):
            out.append((m, "weight"))
    return out


def measure_sparsity(model: nn.Module) -> dict:
    """Return per-layer and global zero-fraction over conv weights."""
    total = 0
    zeros = 0
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, CONV_TYPES):
            w = m.weight.detach()
            n = w.numel()
            z = (w == 0).sum().item()
            layers[name] = {"params": n, "zeros": z, "sparsity": z / max(n, 1)}
            total += n
            zeros += z
    return {
        "global": {"params": total, "zeros": zeros, "sparsity": zeros / max(total, 1)},
        "layers": layers,
    }


def apply_unstructured_magnitude(model: nn.Module, amount: float) -> nn.Module:
    """Global L1 pruning across all conv weights. Permanent (mask removed).

    `amount` is the fraction of *individual weights* zeroed. Pure magnitude.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"amount must be in (0, 1), got {amount}")

    params = _prunable_conv_modules(model)
    if not params:
        raise RuntimeError("No conv modules found in the model")

    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, name in params:
        prune.remove(module, name)
    return model


def _apply_2of4_mask_to_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Return a copy of `weight` with a 2-of-4 zero pattern along its last axis.

    For Conv2d weight (out_c, in_c, kH, kW) we flatten the (in_c, kH, kW) tail
    and zero the 2 smallest-magnitude elements in every consecutive group of 4.
    Tail length is padded with zeros to a multiple of 4 only conceptually — we
    handle the remainder separately so we never resize the actual tensor.
    """
    orig_shape = weight.shape
    flat = weight.detach().reshape(orig_shape[0], -1).clone()
    n_in = flat.size(1)
    n_full = (n_in // 4) * 4

    if n_full > 0:
        body = flat[:, :n_full].reshape(orig_shape[0], -1, 4)
        # Zero the 2 lowest-|x| entries in each group-of-4.
        _, idx = torch.topk(body.abs(), k=2, dim=-1, largest=False)
        mask = torch.ones_like(body)
        mask.scatter_(-1, idx, 0.0)
        body.mul_(mask)
        flat[:, :n_full] = body.reshape(orig_shape[0], n_full)

    # Tail (n_in % 4 < 4 leftover): leave as-is. Strict 2:4 requires divisible.
    return flat.reshape(orig_shape).contiguous()


def apply_2to4_semi_structured(model: nn.Module) -> nn.Module:
    """Apply a 2-of-4 zero pattern to every conv weight (mask-only, in-place).

    Hardware acceleration on Ampere+ requires replacing the layer with one that
    multiplies via a SparseSemiStructuredTensor; this function only enforces
    the *pattern*, so qualitative correctness benchmarks are valid on any GPU.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CONV_TYPES):
                m.weight.data.copy_(_apply_2of4_mask_to_tensor(m.weight.data))
    return model


def _layerwise_2of4_sparsity_ok(model: nn.Module, atol: float = 1e-3) -> bool:
    """Sanity: each conv weight has ~50% zeros in its (out, in*kH*kW) main body."""
    for m in model.modules():
        if isinstance(m, CONV_TYPES):
            w = m.weight.detach()
            n_in = w.shape[1] * w.shape[2] * w.shape[3] if w.dim() == 4 else w.numel() // w.shape[0]
            n_full = (n_in // 4) * 4
            if n_full == 0:
                continue
            body_zeros = (w.reshape(w.shape[0], -1)[:, :n_full] == 0).sum().item()
            if abs(body_zeros / (w.shape[0] * n_full) - 0.5) > atol:
                return False
    return True


def apply_structured_channel_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Mask the lowest-L2-norm output channels of every Conv2d.

    Mask-only: shape is preserved, so this gives a correct quality/sparsity
    measurement but no compute speedup with dense kernels. Physical channel
    removal would require rewiring downstream layers (skip connections in UNet
    make this fragile under time pressure — explicitly out of scope here).
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"amount must be in (0, 1), got {amount}")

    for module, name in _prunable_conv_modules(model):
        prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
        prune.remove(module, name)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
