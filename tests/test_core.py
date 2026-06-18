from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import BinarySegmentationMetrics  # noqa: E402
from model import UNet  # noqa: E402


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(base_channels=8)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1, 64, 64)


def test_metrics_are_one_for_perfect_prediction() -> None:
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metrics = BinarySegmentationMetrics()
    metrics.update(logits, targets)
    result = metrics.compute()
    for value in result.values():
        assert value == pytest.approx(1.0)
