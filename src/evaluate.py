"""Evaluate the best checkpoint on the held-out labeled test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from metrics import BinarySegmentationMetrics
from model import build_model
from utils import get_device, load_checkpoint, load_config, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["random_seed"])
    output_dir = Path(config["paths"]["output_dir"])
    datasets, splits = build_datasets(config)
    loader = DataLoader(
        datasets["test"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    device = get_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint_path = output_dir / "checkpoints" / "best_model.pth"
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    metrics = BinarySegmentationMetrics(config["training"].get("threshold", 0.5))
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            metrics.update(model(images), masks)
    result = metrics.compute()
    metrics_path = output_dir / "metrics.json"
    payload = {}
    if metrics_path.is_file():
        import json

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "test": result,
            "test_samples": len(splits["test"]),
        }
    )
    save_json(payload, metrics_path)
    for name, value in result.items():
        print(f"{name:>16}: {value:.6f}")


if __name__ == "__main__":
    main()
