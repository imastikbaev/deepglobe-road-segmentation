# DeepGlobe Road Segmentation with U-Net

A clean PyTorch pipeline for learning binary road masks from DeepGlobe satellite imagery. It discovers labeled `*_sat.jpg` / `*_mask.png` pairs, creates a deterministic 80/10/10 split, trains U-Net, evaluates the held-out test split, and exports masks and visual comparisons.

## Semantic segmentation, not object detection

Object detection predicts boxes around separate objects. Roads are thin, connected regions with irregular shapes, so boxes are not useful. This project performs **binary semantic segmentation**: every pixel is classified as road (`1`) or background (`0`).

## Dataset

Expected labeled filenames:

```text
train/100034_sat.jpg
train/100034_mask.png
```

The loader searches recursively and pairs files by the shared identifier and directory. Satellite images without matching masks are ignored for training, validation, and evaluation.

The supplied archive contains 6,226 labeled pairs under `train/`. Its `valid/` and `test/` directories are unlabeled, so this project creates its own seeded split from the labeled pairs:

- 80% training
- 10% validation
- 10% test
- seed `42`

The exact split is saved to `outputs/splits.json` and reused while the dataset fingerprint and split settings remain unchanged.

Images are resized to 256×256 with bilinear interpolation and normalized to `[0, 1]`. Masks use nearest-neighbor resizing and are binarized with `pixel >= 128`.

The dataset stays external. `dataset.path` may point either to an extracted directory or directly to `dataset_roads.zip`; do not copy it into this repository.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `configs/config.yaml` if the dataset ZIP or extracted folder is elsewhere:

```yaml
dataset:
  path: "/absolute/path/to/dataset_roads.zip"
```

## Train

```bash
python src/train.py --config configs/config.yaml
```

Resume from the best checkpoint, optionally with a lower learning rate:

```bash
python src/train.py --config configs/config.yaml --resume --learning-rate 0.0005
```

Training uses:

- a four-level U-Net with one-channel logits
- `BCEWithLogitsLoss + Dice Loss`
- horizontal and vertical flips
- paired random rotation (bilinear image, nearest-neighbor mask)
- image-only brightness and contrast augmentation
- AdamW and validation-IoU checkpoint selection
- learning-rate reduction and early stopping

`model.base_channels: 32` is the memory-friendly default. Change it to `64` for the original notebook's channel widths. If GPU memory is insufficient, reduce `training.batch_size` first.

## Evaluate

```bash
python src/evaluate.py --config configs/config.yaml
```

This loads `outputs/checkpoints/best_model.pth`, evaluates the held-out labeled test split, prints all metrics, and writes `outputs/metrics.json`.

## Predict

One image:

```bash
python src/predict.py --config configs/config.yaml --input /path/to/123_sat.jpg
```

A folder (searched recursively):

```bash
python src/predict.py --config configs/config.yaml --input /path/to/images
```

If a matching mask is next to an image, it is included in the visualization. Predictions are written to:

- `outputs/predicted_masks/`
- `outputs/visualizations/`

## Metrics

- **IoU / Jaccard:** road-pixel intersection divided by union.
- **Dice / F1:** overlap score that weights the intersection twice.
- **Precision:** fraction of predicted road pixels that are roads.
- **Recall:** fraction of true road pixels recovered.
- **Pixel accuracy:** fraction of all pixels classified correctly. Because background dominates satellite images, use IoU and Dice as the primary metrics.

Metrics are accumulated from global true-positive, false-positive, false-negative, and true-negative pixel counts after thresholding sigmoid probabilities at `0.5`.

## Baseline training result

The baseline was trained on the reproducible split from all 6,226 labeled pairs. Training was resumed from the first run with a reduced learning rate of `0.0005`. The best checkpoint was selected at epoch 5 by validation IoU and evaluated on the held-out test split of 624 images:

| Metric | Test result |
|---|---:|
| IoU / Jaccard | 0.4520 |
| Dice / F1 | 0.6225 |
| Precision | 0.6152 |
| Recall | 0.6301 |
| Pixel accuracy | 0.9670 |

This run used 256×256 inputs, `base_channels: 32`, batch size 8, threshold 0.5, and an Apple M2 GPU. Pixel accuracy is high partly because background pixels dominate the images; IoU and Dice are the more informative road-segmentation scores. Validation IoU peaked at `0.4527` on epoch 5 and remained below that score on epochs 6 and 7, so the best checkpoint was retained.

## Outputs

```text
outputs/
├── checkpoints/
│   └── best_model.pth
├── metrics.json
├── predicted_masks/
├── splits.json
├── training_history.json
├── training_log.csv
└── visualizations/
```

## Improvements over the original GitHub baseline

The original project was a Colab/Keras notebook that loaded prebuilt HDF5 arrays, used a random Keras validation split, trained with binary cross-entropy alone, and had no reusable evaluation or prediction CLI. This refactor adds direct DeepGlobe pairing, robust mask binarization, deterministic train/validation/test manifests, lazy directory-or-ZIP loading, paired augmentation, logits-based BCE plus Dice loss, complete metrics, best-checkpoint selection, structured logs, standalone evaluation, and batch prediction.

## Troubleshooting

Check these first if training fails:

1. Confirm `dataset.path` exists and points to the dataset root or ZIP.
2. Confirm labeled pairs use exactly `*_sat.jpg` and `*_mask.png`.
3. Reduce `batch_size` if CUDA/MPS reports out-of-memory.
4. Leave `num_workers: 0` initially, especially when reading directly from ZIP or on Windows.
5. Delete `outputs/splits.json` after intentionally changing dataset contents, although fingerprint changes are normally detected automatically.
6. Verify your PyTorch build supports the selected device; set `device: cpu` as a diagnostic.
