from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filter_potholes import (  # noqa: E402
    calculate_road_overlap,
    create_visualization,
    filter_pothole_detections,
    normalized_bbox_to_pixels,
)
from mock_predictions import MOCK_PREDICTIONS  # noqa: E402


def pothole(**extra):
    detection = {
        "bbox": [100, 100, 200, 200],
        "class_code": "D40",
        "confidence": 0.54,
        "class_label": "Яма",
        "bbox_normalized": [0.25, 0.25, 0.75, 0.75],
    }
    detection.update(extra)
    return detection


def test_normalized_bbox_to_pixels() -> None:
    assert normalized_bbox_to_pixels([0.25, 0.2, 0.75, 0.8], 200, 100) == (
        50,
        20,
        150,
        80,
    )


def test_bbox_on_image_boundary_is_clipped() -> None:
    assert normalized_bbox_to_pixels([-0.2, -0.1, 1.2, 1.4], 256, 256) == (
        0,
        0,
        256,
        256,
    )
    assert normalized_bbox_to_pixels([0.99, 0.99, 1.1, 1.1], 256, 256) == (
        253,
        253,
        256,
        256,
    )


def test_tiny_bbox_has_at_least_one_pixel() -> None:
    bbox = normalized_bbox_to_pixels(
        [0.50001, 0.50001, 0.50002, 0.50002], 256, 256
    )
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x2 - x1 >= 1
    assert y2 - y1 >= 1


def test_full_road_overlap() -> None:
    mask = np.ones((4, 4), dtype=bool)
    assert calculate_road_overlap(mask, (1, 1, 3, 3)) == pytest.approx(1.0)


def test_no_road_overlap() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    assert calculate_road_overlap(mask, (1, 1, 3, 3)) == pytest.approx(0.0)


def test_partial_road_overlap() -> None:
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    assert calculate_road_overlap(mask, (0, 0, 2, 2)) == pytest.approx(0.5)


def test_d00_and_d10_are_excluded() -> None:
    mask = np.ones((8, 8), dtype=bool)
    detections = [
        pothole(class_code="D00", class_label="Продольная трещина"),
        pothole(class_code="D10", class_label="Поперечная трещина"),
    ]
    accepted, rejected = filter_pothole_detections(detections, mask)
    assert accepted == []
    assert rejected == []


def test_original_fields_are_preserved() -> None:
    mask = np.ones((8, 8), dtype=bool)
    source = pothole(detector_id="abc-123", metadata={"source": "mock"})
    accepted, rejected = filter_pothole_detections([source], mask)
    assert rejected == []
    assert len(accepted) == 1
    for key, value in source.items():
        assert accepted[0][key] == value
    assert accepted[0]["road_overlap"] == pytest.approx(1.0)
    assert "road_overlap" not in source


def test_rejected_detection_keeps_overlap() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    accepted, rejected = filter_pothole_detections([pothole()], mask)
    assert accepted == []
    assert rejected[0]["road_overlap"] == pytest.approx(0.0)


def test_missing_image_in_mock_predictions() -> None:
    assert MOCK_PREDICTIONS.get("missing.jpg", []) == []


def test_empty_or_outside_bbox_is_rejected() -> None:
    mask = np.ones((8, 8), dtype=bool)
    detections = [
        pothole(bbox_normalized=[0.5, 0.5, 0.5, 0.8]),
        pothole(bbox_normalized=[1.1, 0.1, 1.2, 0.2]),
    ]
    accepted, rejected = filter_pothole_detections(detections, mask)
    assert accepted == []
    assert len(rejected) == 2


def test_missing_or_malformed_bbox_is_rejected_without_crashing() -> None:
    mask = np.ones((8, 8), dtype=bool)
    missing = pothole()
    missing.pop("bbox_normalized")
    malformed = pothole(bbox_normalized=[0.1, 0.2, 0.3])
    accepted, rejected = filter_pothole_detections([missing, malformed], mask)
    assert accepted == []
    assert len(rejected) == 2
    assert all(item["road_overlap"] == 0.0 for item in rejected)
    visualization = create_visualization(
        Image.new("RGB", (16, 16), "black"), mask, [], rejected
    )
    assert visualization.size == (16, 16)
