"""Local web application for interactive DeepGlobe road segmentation."""

from __future__ import annotations

import argparse
import base64
import io
import time
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from model import build_model
from utils import get_device, load_checkpoint, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "web"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class ModelService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = load_config(str(config_path))
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.checkpoint_path = self.output_dir / "checkpoints" / "best_model.pth"
        self.device = get_device(self.config["training"].get("device", "auto"))
        self.model = None
        self.checkpoint_epoch = None

    @property
    def ready(self) -> bool:
        return self.checkpoint_path.is_file()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.ready:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        model = build_model(self.config).to(self.device)
        checkpoint = load_checkpoint(model, self.checkpoint_path, self.device)
        model.eval()
        self.model = model
        self.checkpoint_epoch = checkpoint.get("epoch")

    @torch.inference_mode()
    def predict(self, image: Image.Image, threshold: float) -> np.ndarray:
        self.load()
        image_size = self.config["dataset"].get("image_size", 256)
        resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        return (probability >= threshold).astype(np.uint8)


def encode_mask(mask: np.ndarray) -> str:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask > 0] = (255, 93, 67, 105)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def remove_small_components(mask: np.ndarray, minimum_fraction: float = 0.0015) -> np.ndarray:
    minimum_area = max(16, int(mask.size * minimum_fraction))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == component] = 1
    return cleaned


def mask_to_contours(mask: np.ndarray) -> List[List[List[float]]]:
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    result = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 2:
            continue
        epsilon = max(0.8, 0.006 * cv2.arcLength(contour, True))
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        points = [[float(x) / width, float(y) / height] for x, y in simplified[:500]]
        result.append(points)
        if len(result) >= 500:
            break
    return result


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    app = FastAPI(title="RoadLens", version="1.0")
    service = ModelService(config_path)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/styles.css")
    def styles():
        return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    def javascript():
        return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")

    @app.get("/api/status")
    def status():
        return {
            "ready": service.ready,
            "device": str(service.device),
            "checkpoint": service.checkpoint_path.name if service.ready else None,
            "checkpoint_epoch": service.checkpoint_epoch,
            "image_size": service.config["dataset"].get("image_size", 256),
        }

    @app.get("/api/demo")
    def demo():
        dataset_path = Path(service.config["dataset"]["path"])
        if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(dataset_path) as archive:
                names = set(archive.namelist())
                labeled_candidates = sorted(
                    name
                    for name in names
                    if name.lower().endswith("_sat.jpg")
                    and name[:-8] + "_mask.png" in names
                )
                preferred = "test/7890_sat.jpg"
                candidates = [preferred] if preferred in names else labeled_candidates
                if not candidates:
                    raise HTTPException(404, "No demo image was found in the dataset ZIP.")
                filename = Path(candidates[0]).name
                contents = archive.read(candidates[0])
        elif dataset_path.is_dir():
            candidates = sorted(dataset_path.rglob("*_sat.jpg"))
            image_path = next(
                (
                    path
                    for path in candidates
                    if path.with_name(path.name[:-8] + "_mask.png").is_file()
                ),
                None,
            )
            if image_path is None:
                raise HTTPException(404, "No labeled demo image was found in the dataset directory.")
            filename = image_path.name
            contents = image_path.read_bytes()
        else:
            raise HTTPException(404, "Configured dataset path is unavailable.")
        return Response(
            contents,
            media_type="image/jpeg",
            headers={"X-Demo-Filename": filename},
        )

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        threshold: float = Form(0.5),
    ):
        if not 0.05 <= threshold <= 0.95:
            raise HTTPException(400, "Threshold must be between 0.05 and 0.95.")
        contents = await image.read()
        if not contents:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Maximum upload size is 25 MB.")
        try:
            source = Image.open(io.BytesIO(contents)).convert("RGB")
            source.load()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(400, "Please upload a valid JPG, PNG, TIFF, or WebP image.") from error
        if source.width < 32 or source.height < 32:
            raise HTTPException(400, "Image dimensions must be at least 32×32 pixels.")

        started = time.perf_counter()
        try:
            mask = service.predict(source, threshold)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        mask = remove_small_components(mask)
        contours = mask_to_contours(mask)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "filename": image.filename,
            "width": source.width,
            "height": source.height,
            "threshold": threshold,
            "road_coverage": round(float(mask.mean()) * 100, 2),
            "contour_count": len(contours),
            "inference_ms": elapsed_ms,
            "checkpoint_epoch": service.checkpoint_epoch,
            "mask_overlay": encode_mask(mask),
            "contours": contours,
        }

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local RoadLens web platform.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(Path(args.config).expanduser().resolve())
    uvicorn.run(app, host=args.host, port=args.port)
