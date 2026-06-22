"""Filter mock pothole detections using the U-Net road segmentation mask."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mock_predictions import MOCK_PREDICTIONS  # noqa: E402
from model import build_model  # noqa: E402
from utils import get_device, load_checkpoint, load_config  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PixelBBox = tuple[int, int, int, int]
Detection = dict[str, Any]


def collect_images(input_path: Path) -> list[Path]:
    """Return supported images from a file or directory in stable order."""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def predict_road_probability(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run road segmentation and return a sigmoid probability map."""
    resized = image.convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(array.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return probability.astype(np.float32, copy=False)


def normalized_bbox_to_pixels(
    bbox_normalized: Sequence[float],
    width: int,
    height: int,
) -> Optional[PixelBBox]:
    """Map normalized xyxy coordinates to a clipped, non-empty pixel bbox.

    The returned coordinates use NumPy/Pillow slicing semantics:
    ``x1``/``y1`` are inclusive and ``x2``/``y2`` are exclusive. Positive
    sub-pixel boxes are expanded to at least one pixel in each dimension.
    Boxes with no intersection with the image or non-positive source area are
    rejected with ``None``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if len(bbox_normalized) != 4:
        raise ValueError("bbox_normalized must contain [x1, y1, x2, y2]")

    try:
        x1n, y1n, x2n, y2n = (float(value) for value in bbox_normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox_normalized values must be numeric") from error
    if not all(math.isfinite(value) for value in (x1n, y1n, x2n, y2n)):
        raise ValueError("bbox_normalized values must be finite")
    if x2n <= x1n or y2n <= y1n:
        return None
    if x2n <= 0.0 or y2n <= 0.0 or x1n >= 1.0 or y1n >= 1.0:
        return None

    x1n = min(max(x1n, 0.0), 1.0)
    y1n = min(max(y1n, 0.0), 1.0)
    x2n = min(max(x2n, 0.0), 1.0)
    y2n = min(max(y2n, 0.0), 1.0)

    x1 = min(max(math.floor(x1n * width), 0), width - 1)
    y1 = min(max(math.floor(y1n * height), 0), height - 1)
    x2 = min(max(math.ceil(x2n * width), 1), width)
    y2 = min(max(math.ceil(y2n * height), 1), height)

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def calculate_road_overlap(
    road_mask: np.ndarray,
    pixel_bbox: Optional[PixelBBox],
) -> float:
    """Return the fraction of road pixels inside a clipped pixel bbox."""
    if road_mask.ndim != 2:
        raise ValueError("road_mask must be a two-dimensional array")
    if pixel_bbox is None:
        return 0.0

    height, width = road_mask.shape
    x1, y1, x2, y2 = pixel_bbox
    x1 = min(max(int(x1), 0), width)
    y1 = min(max(int(y1), 0), height)
    x2 = min(max(int(x2), 0), width)
    y2 = min(max(int(y2), 0), height)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = np.asarray(road_mask[y1:y2, x1:x2], dtype=bool)
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region) / region.size)


def is_pothole_detection(detection: Detection) -> bool:
    """Identify potholes while supporting code- and label-based payloads."""
    return (
        detection.get("class_code") == "D40"
        or detection.get("class_label") == "Яма"
    )


def detection_bbox_to_pixels(
    detection: Detection,
    width: int,
    height: int,
) -> Optional[PixelBBox]:
    """Safely read and map a detection's normalized bbox."""
    bbox = detection.get("bbox_normalized")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        return None
    try:
        return normalized_bbox_to_pixels(bbox, width, height)
    except (TypeError, ValueError):
        return None


def filter_pothole_detections(
    detections: Iterable[Detection],
    road_mask: np.ndarray,
    min_road_overlap: float = 0.5,
) -> tuple[list[Detection], list[Detection]]:
    """Split D40 detections into accepted and rejected lists.

    D00, D10, and other detector classes are ignored. Every accepted/rejected
    item is a shallow copy that preserves all source fields and adds only
    ``road_overlap``.
    """
    if not 0.0 <= min_road_overlap <= 1.0:
        raise ValueError("min_road_overlap must be between 0 and 1")

    height, width = road_mask.shape
    accepted: list[Detection] = []
    rejected: list[Detection] = []
    for detection in detections:
        if not is_pothole_detection(detection):
            continue
        pixel_bbox = detection_bbox_to_pixels(detection, width, height)
        overlap = calculate_road_overlap(road_mask, pixel_bbox)
        enriched = dict(detection)
        enriched["road_overlap"] = overlap
        if pixel_bbox is not None and overlap >= min_road_overlap:
            accepted.append(enriched)
        else:
            rejected.append(enriched)
    return accepted, rejected


def _draw_detection(
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    image_size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    pixel_bbox = detection_bbox_to_pixels(
        detection, image_size[0], image_size[1]
    )
    if pixel_bbox is None:
        return
    x1, y1, x2, y2 = pixel_bbox
    line_width = max(3, round(max(image_size) / 800))
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=line_width)

    confidence = detection.get("confidence")
    confidence_text = (
        f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "n/a"
    )
    label = (
        f"conf {confidence_text} | road "
        f"{float(detection.get('road_overlap', 0.0)):.2f}"
    )
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_y = max(0, y1 - text_height - 8)
    draw.rectangle(
        (x1, label_y, min(image_size[0], x1 + text_width + 8), label_y + text_height + 8),
        fill=color,
    )
    draw.text((x1 + 4, label_y + 4), label, fill="white", font=font)


def create_visualization(
    image: Image.Image,
    road_mask: np.ndarray,
    accepted: Iterable[Detection],
    rejected: Iterable[Detection],
) -> Image.Image:
    """Overlay the road mask and accepted/rejected pothole detections."""
    original = image.convert("RGB")
    mask_image = Image.fromarray(
        np.asarray(road_mask, dtype=np.uint8) * 255, mode="L"
    ).resize(original.size, Image.Resampling.NEAREST)

    base = original.convert("RGBA")
    green = Image.new("RGBA", original.size, (0, 255, 0, 0))
    green.putalpha(mask_image.point(lambda value: 80 if value else 0))
    visualization = Image.alpha_composite(base, green).convert("RGB")
    draw = ImageDraw.Draw(visualization)

    for detection in accepted:
        _draw_detection(draw, detection, original.size, (0, 190, 0))
    for detection in rejected:
        _draw_detection(draw, detection, original.size, (220, 0, 0))
    return visualization


class RoadPotholeFilter:
    """Load road segmentation once and process multiple images."""

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        self.config = load_config(config_path)
        self.device = get_device(self.config["training"].get("device", "auto"))
        self.model = build_model(self.config).to(self.device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        load_checkpoint(self.model, self.checkpoint_path, self.device)
        self.model.eval()
        self.image_size = int(self.config["dataset"].get("image_size", 256))
        self.threshold = float(self.config["training"].get("threshold", 0.5))

    def predict_mask(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        probability = predict_road_probability(
            self.model, image, self.image_size, self.device
        )
        road_mask = probability >= self.threshold
        return probability, road_mask


def run_filtering(
    filter_model: RoadPotholeFilter,
    input_path: Path,
    output_dir: Path,
    min_road_overlap: float,
    predictions: dict[str, list[Detection]],
) -> tuple[dict[str, list[Detection]], dict[str, dict[str, int]]]:
    """Process all images and save JSON, masks, and visualizations."""
    images = collect_images(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "road_masks"
    visualization_dir = output_dir / "visualizations"
    mask_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    filtered_predictions: dict[str, list[Detection]] = {}
    counts: dict[str, dict[str, int]] = {}
    for image_path in images:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        _, road_mask = filter_model.predict_mask(image)
        detections = predictions.get(image_path.name, [])
        accepted, rejected = filter_pothole_detections(
            detections, road_mask, min_road_overlap
        )
        filtered_predictions[image_path.name] = accepted
        counts[image_path.name] = {
            "before": sum(1 for item in detections if is_pothole_detection(item)),
            "after": len(accepted),
        }

        full_size_mask = Image.fromarray(
            road_mask.astype(np.uint8) * 255, mode="L"
        ).resize(image.size, Image.Resampling.NEAREST)
        full_size_mask.save(mask_dir / f"{image_path.stem}_road_mask.png")

        visualization = create_visualization(image, road_mask, accepted, rejected)
        visualization.save(
            visualization_dir / f"{image_path.stem}_filtered.jpg",
            quality=95,
        )
        print(
            f"{image_path.name}: D40 {counts[image_path.name]['before']} -> "
            f"{counts[image_path.name]['after']}"
        )

    json_path = output_dir / "filtered_predictions.json"
    json_path.write_text(
        json.dumps(filtered_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filtered_predictions, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep mock D40 potholes that overlap the segmented road."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--min-road-overlap", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/filtered")
    args = parser.parse_args()

    filter_model = RoadPotholeFilter(args.config, args.checkpoint)
    run_filtering(
        filter_model=filter_model,
        input_path=Path(args.input).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        min_road_overlap=args.min_road_overlap,
        predictions=MOCK_PREDICTIONS,
    )


if __name__ == "__main__":
    main()
