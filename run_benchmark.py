import json

import torch

from src.dataset import CarvanaTrainDataset
from src.experiment import ExperimentConfig, UNetSegmentationExperiments


train_root = "/content/drive/MyDrive/carvana/train"
mask_root = "/content/drive/MyDrive/carvana/train_masks"

n_classes = 2
image_scale = 0.5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Creating Carvana train image+mask dataset...")
dataset = CarvanaTrainDataset(
    train_root=train_root,
    mask_root=mask_root,
    scale=image_scale,
)

experiments = UNetSegmentationExperiments(
    dataset=dataset,
    device=device,
    num_images=1280,
    num_classes=n_classes,
)

configs = [
    ExperimentConfig(batch_size=2, method="normal_fp16"),
    ExperimentConfig(batch_size=2, method="torch_compile"),
    ExperimentConfig(batch_size=4, method="normal_fp16"),
    ExperimentConfig(batch_size=4, method="torch_compile"),
    ExperimentConfig(batch_size=8, method="normal_fp16"),
    ExperimentConfig(batch_size=8, method="torch_compile"),
    ExperimentConfig(batch_size=16, method="normal_fp16"),
    ExperimentConfig(batch_size=16, method="torch_compile"),
    ExperimentConfig(batch_size=32, method="normal_fp16"),
    ExperimentConfig(batch_size=32, method="torch_compile"),
]

results = []
for cfg in configs:
    print(f"\nRunning config: {cfg}")
    result = experiments.make_experiment(cfg)
    results.append(result)
    print("Result:")
    print(json.dumps(result, indent=2))

print("\nCollected history:")
print(json.dumps(experiments.history, indent=2))

output_path = "unet_benchmarks.json"
experiments.save_history(output_path)

with open(output_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)

print("Loaded history from disk:")
print(json.dumps(loaded, indent=2))
