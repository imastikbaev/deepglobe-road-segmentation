# U-Net Road Segmentation: DeepGlobe → UAVid

Binary semantic segmentation of roads in satellite and UAV imagery using a
four-level U-Net. The released checkpoint uses `base_channels=32`: it was
pretrained on DeepGlobe and fine-tuned on binary road masks derived from UAVid.

## Released model

**[Download `best_model_uavid_finetuned_iou7168.pth`](https://github.com/imastikbaev/deepglobe-road-segmentation/releases/download/v1.0-uavid/best_model_uavid_finetuned_iou7168.pth)**

Release: [UAVid Fine-tuned Road Segmentation Model](https://github.com/imastikbaev/deepglobe-road-segmentation/releases/tag/v1.0-uavid)

| Metric | UAVid test score |
|---|---:|
| IoU / Jaccard | **0.7168** |
| Dice / F1 | **0.8350** |
| Precision | **0.8149** |
| Recall | **0.8561** |
| Pixel accuracy | **0.9550** |
| BCE + Dice loss | **0.1963** |

The best checkpoint was selected at epoch 47 with validation IoU `0.6998` and
validation Dice `0.8234`. IoU and Dice are the primary quality indicators;
pixel accuracy is less informative because background pixels dominate the data.

See [MODEL_CARD.md](MODEL_CARD.md) for training details, intended use, and
limitations. Machine-readable results are in
[results/test_metrics.json](results/test_metrics.json).

## Features

- deterministic train/validation/test splits;
- paired image and mask augmentation;
- BCE + Dice training loss;
- IoU, Dice, precision, recall, and pixel-accuracy evaluation;
- checkpoint resume and learning-rate override;
- prediction for one image or an entire folder;
- CPU, CUDA, and Apple Silicon MPS support.

This is semantic segmentation, not object detection: every pixel is classified
as road or background.

## Repository structure

```text
.
├── configs/
│   ├── config.yaml
│   └── config.uavid_finetune.yaml
├── results/
│   └── test_metrics.json
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── filter_potholes.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
├── mock_predictions.py
├── MODEL_CARD.md
├── requirements.txt
└── requirements-dev.txt
```

Model weights, datasets, generated predictions, and other large artifacts are
excluded from Git by `.gitignore`.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/imastikbaev/deepglobe-road-segmentation.git
cd deepglobe-road-segmentation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Dataset preparation

The loader expects paired files:

```text
sample_sat.jpg
sample_mask.png
```

For the released model, UAVid road-class pixels (`RGB 128, 64, 128`) were
converted to binary masks and saved in this format. The resulting 670 labeled
pairs were split deterministically with seed 42:

| Split | Images |
|---|---:|
| Train | 536 |
| Validation | 67 |
| Test | 67 |

Place the prepared data outside Git, for example at
`data/uavid_binary_roads/`. The dataset directory and archives are ignored.

## Use the released checkpoint

Download the checkpoint:

```bash
mkdir -p outputs/uavid/checkpoints
curl -L \
  https://github.com/imastikbaev/deepglobe-road-segmentation/releases/download/v1.0-uavid/best_model_uavid_finetuned_iou7168.pth \
  -o outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth
```

SHA-256:

```text
32fd1735b805f27bf31048f14a276ff3c3d5cba78b7ca2f63c6d49ea6dea1468
```

## Prediction

```bash
python src/predict.py \
  --config configs/config.uavid_finetune.yaml \
  --checkpoint outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth \
  --input path/to/image-or-folder
```

Predicted masks and comparison images are written to:

```text
outputs/uavid/predicted_masks/
outputs/uavid/visualizations/
```

## Evaluation

After setting `dataset.path` in `configs/config.uavid_finetune.yaml`:

```bash
python src/evaluate.py \
  --config configs/config.uavid_finetune.yaml \
  --checkpoint outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth
```

## Filter pothole detections by the road mask

`src/filter_potholes.py` integrates the road-segmentation model with detector
results from [`mock_predictions.py`](mock_predictions.py). It keeps only
potholes (`class_code == "D40"` or `class_label == "Яма"`) whose bounding box
contains enough segmented road pixels.

Detector payloads must be keyed by the exact image filename:

```python
MOCK_PREDICTIONS = {
    "83.jpg": [
        {
            "bbox": [x1, y1, x2, y2],
            "class_code": "D40",
            "confidence": 0.54,
            "class_label": "Яма",
            "bbox_normalized": [x1_norm, y1_norm, x2_norm, y2_norm],
        }
    ]
}
```

All original detector fields are preserved. The filter adds:

- `confidence`: the detector's confidence, copied without modification;
- `road_overlap`: road pixels inside the normalized bbox divided by all pixels
  inside that bbox on the model's `256×256` binary mask.

`bbox_normalized` is required for overlap and visualization. It remains correct
when detector coordinates refer to `4000×3000` but the loaded image has another
resolution, such as `2048×1536`. The pixel-valued `bbox` is preserved in JSON
but is not used for coordinate mapping.

Run the integration:

```bash
python src/filter_potholes.py \
  --config configs/config.uavid_finetune.yaml \
  --checkpoint outputs/uavid/checkpoints/best_model_uavid_finetuned_iou7168.pth \
  --input path/to/images \
  --min-road-overlap 0.5 \
  --output-dir outputs/filtered
```

`--min-road-overlap` accepts values from `0` to `1` and defaults to `0.5`.
The model and configuration are loaded once, then all input images are
processed directly in Python.

Outputs:

```text
outputs/filtered/
├── filtered_predictions.json
├── road_masks/
│   └── 83_road_mask.png
└── visualizations/
    └── 83_filtered.jpg
```

Road masks are saved at each source image's original resolution using nearest
neighbor resizing. Visualizations contain a translucent green road mask, green
boxes for accepted D40 detections, and red boxes for rejected D40 detections.
D00 and D10 detections are not displayed.

`mock_predictions.py` is an integration fixture. In production, replace its
dictionary lookup with the real defect-detector call while keeping the same
detection schema and filtering functions.

## Reproduce fine-tuning

The released run resumed from a DeepGlobe checkpoint and used a learning rate
of `5e-5`:

```bash
python src/train.py \
  --config configs/config.uavid_finetune.yaml \
  --resume path/to/deepglobe_checkpoint.pth \
  --learning-rate 0.00005
```

Training outputs are saved under `outputs/uavid/`.

## Development checks

```bash
pip install -r requirements-dev.txt
python -m compileall -q src
pytest
```

## Limitations

- Inputs are resized to `256×256`, which can remove narrow-road detail.
- Results may degrade across sensors, regions, seasons, and flight altitudes.
- Masks do not guarantee topological road connectivity.
- The model is not intended for safety-critical routing or autonomous driving.

## License

MIT. See [LICENSE](LICENSE).
