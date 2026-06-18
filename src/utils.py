"""Shared configuration, reproducibility, device, and checkpoint helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    project_root = path.parent.parent
    dataset_path = Path(str(config["dataset"]["path"])).expanduser()
    output_dir = Path(str(config.get("paths", {}).get("output_dir", "outputs"))).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    config["dataset"]["path"] = str(dataset_path)
    config.setdefault("paths", {})["output_dir"] = str(output_dir)
    config["_project_root"] = str(project_root)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\nRun src/train.py first or update paths.output_dir."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
