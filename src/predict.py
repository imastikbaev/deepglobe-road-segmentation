"""Predict road masks for one image or a folder and save visual comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import build_model
from utils import get_device, load_checkpoint, load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES and "_mask" not in path.stem.lower()
        )
    raise FileNotFoundError(f"Prediction input does not exist: {input_path}")


def matching_mask(image_path: Path) -> Optional[Path]:
    if image_path.name.lower().endswith("_sat.jpg"):
        candidate = image_path.with_name(image_path.name[:-8] + "_mask.png")
        if candidate.is_file():
            return candidate
    return None


def panel(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), title, fill="black")
    return canvas


def save_visualization(image: Image.Image, prediction: Image.Image, ground_truth: Optional[Image.Image], path: Path):
    panels = [panel(image, "Satellite")]
    if ground_truth is not None:
        panels.append(panel(ground_truth, "Ground truth"))
    panels.append(panel(prediction, "Prediction"))
    canvas = Image.new("RGB", (sum(item.width for item in panels), panels[0].height), "white")
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="Image file or folder")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])
    mask_dir = output_dir / "predicted_masks"
    vis_dir = output_dir / "visualizations"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(model, output_dir / "checkpoints" / "best_model.pth", device)
    model.eval()
    image_size = config["dataset"].get("image_size", 256)
    threshold = config["training"].get("threshold", 0.5)
    images = collect_images(Path(args.input).expanduser())
    if not images:
        raise RuntimeError("No supported input images were found.")

    for image_path in images:
        original = Image.open(image_path).convert("RGB")
        resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        binary = (probability >= threshold).astype(np.uint8) * 255
        predicted = Image.fromarray(binary, mode="L")
        output_name = image_path.stem.replace("_sat", "") + "_pred.png"
        predicted.save(mask_dir / output_name)

        mask_path = matching_mask(image_path)
        ground_truth = None
        if mask_path:
            ground_truth = Image.open(mask_path).convert("L").resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
            ground_truth = Image.fromarray(
                (np.asarray(ground_truth, dtype=np.uint8) >= config["dataset"].get("mask_threshold", 128)).astype(
                    np.uint8
                )
                * 255
            )
        save_visualization(resized, predicted, ground_truth, vis_dir / output_name)
        print(f"Saved {mask_dir / output_name}")


if __name__ == "__main__":
    main()
