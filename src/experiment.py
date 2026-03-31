import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from src.model import load_pretrained_unet
from src.dataset import build_dataloader
from src.benchmark import run_inference_benchmark


@dataclass
class ExperimentConfig:
    batch_size: int
    method: str  # "normal_fp16" or "torch_compile" for now

    def to_dict(self) -> dict:
        return {"batch_size": self.batch_size, "method": self.method}

    def to_json_key(self) -> str:
        # Stable, sorted JSON representation suitable as a dict key.
        return json.dumps(self.to_dict(), sort_keys=True)


class UNetSegmentationExperiments:
    """Orchestrate UNet segmentation inference benchmarks using configs.

    Instead of "number of batches", this class is configured by the
    "number of images" to process in each experiment. The effective
    number of batches is computed per-config based on its batch size.
    """

    def __init__(
        self,
        dataset: Dataset,
        device: Optional[torch.device] = None,
        num_images: int = 320,
        num_classes: int = 2,
    ) -> None:
        self.dataset = dataset
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.num_images = num_images
        self.num_classes = num_classes
        self.history: dict[str, dict] = {}

    def _prepare_base_model(self, scale: float = 0.5):
        """Load a fresh base UNet model on the configured device."""
        model = load_pretrained_unet(scale=scale, device=self.device)
        return model

    def make_experiment(self, config: ExperimentConfig, scale: float = 0.5) -> dict:
        """Run a single experiment given an ExperimentConfig.

        Builds a fresh DataLoader using the config batch size and dispatches
        to the appropriate inference method.
        """
        if config.method not in {"normal_fp16", "torch_compile"}:
            raise ValueError(
                f"Unsupported method '{config.method}'. Expected 'normal_fp16' or 'torch_compile'."
            )

        dataloader = build_dataloader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=2,
            shuffle=False,
        )

        # Compute how many batches are needed to process approximately
        # `self.num_images` images for this batch size.
        effective_num_batches = max(1, math.ceil(self.num_images / config.batch_size))

        if config.method == "normal_fp16":
            result = self._run_normal_fp16(
                dataloader=dataloader,
                num_batches=effective_num_batches,
                scale=scale,
            )
        else:  # "torch_compile"
            result = self._run_torch_compile(
                dataloader=dataloader,
                num_batches=effective_num_batches,
                scale=scale,
            )

        # Attach config metadata
        if not result.get("skipped"):
            result["config"] = config.to_dict()
            result["method"] = config.method
        else:
            # Even for skipped runs, include config metadata
            result.setdefault("config", config.to_dict())
            result.setdefault("method", config.method)

        key = config.to_json_key()
        self.history[key] = result
        return result

    def _run_normal_fp16(self, dataloader: DataLoader, num_batches: int, scale: float = 0.5) -> dict:
        """Internal helper for the normal fp16-style baseline experiment."""
        print("Running experiment: normal_fp16 (fp16 baseline)")
        model = self._prepare_base_model(scale=scale)

        cuda_available = self.device.type == "cuda" and torch.cuda.is_available()
        mixed_precision = False

        if cuda_available:
            model = model.half()
            mixed_precision = True
            print("CUDA available: using fp16 weights + autocast for inference.")
        else:
            print("CUDA not available: running in fp32 without mixed precision.")

        result = run_inference_benchmark(
            model,
            dataloader,
            device=self.device,
            num_batches=num_batches,
            mixed_precision=mixed_precision,
            num_classes=self.num_classes,
        )
        print(f"Finished normal_fp16: {result}")
        return result

    def _run_torch_compile(self, dataloader: DataLoader, num_batches: int, scale: float = 0.5) -> dict:
        """Internal helper for the torch.compile experiment."""
        print("Running experiment: torch_compile")

        if not hasattr(torch, "compile"):
            reason = "torch.compile is not available in this PyTorch version."
            print(reason)
            return {"skipped": True, "reason": reason}

        model = self._prepare_base_model(scale=scale)

        cuda_available = self.device.type == "cuda" and torch.cuda.is_available()
        mixed_precision = False

        if cuda_available:
            model = model.half()
            mixed_precision = True
            print("CUDA available: using fp16 weights + autocast for compiled model.")
        else:
            print("CUDA not available: running compiled model in fp32.")

        try:
            compiled_model = torch.compile(model)
        except Exception as e:
            reason = f"torch.compile failed: {e}"
            print(reason)
            return {"skipped": True, "reason": reason}

        result = run_inference_benchmark(
            compiled_model,
            dataloader,
            device=self.device,
            num_batches=num_batches,
            mixed_precision=mixed_precision,
            num_classes=self.num_classes,
        )
        print(f"Finished torch_compile: {result}")
        return result

    def save_history(self, path: str) -> None:
        """Save experiment history to a JSON file at the given path.

        Overwrites existing files.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")
