from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_metrics_match_uavid_config() -> None:
    metrics = json.loads(
        (PROJECT_ROOT / "results" / "test_metrics.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "config.uavid_finetune.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert metrics["release"] == "v1.0-uavid"
    assert metrics["base_channels"] == config["model"]["base_channels"] == 32
    assert metrics["image_size"] == config["dataset"]["image_size"] == 256
    assert metrics["prediction_threshold"] == config["training"]["threshold"] == 0.5
    assert metrics["best_epoch"] == 47
    assert sum(metrics["split_counts"].values()) == 670
    assert metrics["test"] == {
        "iou": 0.7168,
        "dice": 0.8350,
        "precision": 0.8149,
        "recall": 0.8561,
        "pixel_accuracy": 0.9550,
        "loss": 0.1963,
    }
