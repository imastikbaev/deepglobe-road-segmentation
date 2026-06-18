"""Train U-Net on reproducibly split DeepGlobe labeled pairs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from metrics import BCEDiceLoss, BinarySegmentationMetrics
from model import build_model
from utils import get_device, load_config, save_json, set_seed


def make_loader(dataset, config: dict, shuffle: bool) -> DataLoader:
    training = config["training"]
    generator = torch.Generator().manual_seed(config["random_seed"])
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=training.get("num_workers", 0) > 0,
        generator=generator,
    )


def run_epoch(model, loader, criterion, device, threshold, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = BinarySegmentationMetrics(threshold)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if training else "validate"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            metrics.update(logits.detach(), masks)
    result = metrics.compute()
    result["loss"] = total_loss / max(len(loader.dataset), 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["random_seed"])

    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    datasets, splits = build_datasets(config)
    loaders = {
        name: make_loader(dataset, config, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    print("Labeled split sizes:", {name: len(items) for name, items in splits.items()})

    device = get_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")
    model = build_model(config).to(device)
    loss_cfg = config.get("loss", {})
    criterion = BCEDiceLoss(loss_cfg.get("bce_weight", 0.5), loss_cfg.get("dice_weight", 0.5))
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    threshold = config["training"].get("threshold", 0.5)
    monitor = config["training"].get("monitor", "iou")
    patience = config["training"].get("early_stopping_patience", 10)
    best_score, stale_epochs = -1.0, 0
    best_path = checkpoint_dir / "best_model.pth"
    history = []
    log_path = output_dir / "training_log.csv"

    for epoch in range(1, config["training"]["epochs"] + 1):
        start = time.time()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, threshold, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device, threshold)
        scheduler.step(val_metrics[monitor])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 2),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val IoU {val_metrics['iou']:.4f} | "
            f"val Dice {val_metrics['dice']:.4f}"
        )
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        score = val_metrics[monitor]
        if score > best_score:
            best_score, stale_epochs = score, 0
            torch.save(
                {
                    "epoch": epoch,
                    "monitor": monitor,
                    "best_score": best_score,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "validation_metrics": val_metrics,
                },
                best_path,
            )
            print(f"Saved new best checkpoint: {best_path}")
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device, threshold)
    final = {
        "best_epoch": checkpoint["epoch"],
        "monitor": monitor,
        "best_validation_score": checkpoint["best_score"],
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "checkpoint": str(best_path),
    }
    save_json(final, output_dir / "metrics.json")
    save_json(history, output_dir / "training_history.json")
    print("Test metrics:", {key: round(value, 4) for key, value in test_metrics.items()})


if __name__ == "__main__":
    main()
