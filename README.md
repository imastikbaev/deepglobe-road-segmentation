# DeepGlobe Road Segmentation

U-Net pipeline for detecting roads in satellite images with binary semantic segmentation.

The repository includes dataset loading, training, evaluation, and batch prediction for a clean model-only workflow.

## What it does

- pairs DeepGlobe files named `*_sat.jpg` and `*_mask.png`;
- ignores images without masks during training;
- creates a reproducible 80/10/10 train/validation/test split;
- trains a four-level U-Net with BCE + Dice loss;
- reports IoU, Dice, precision, recall, and pixel accuracy;
- predicts masks for individual images or folders.

This is **semantic segmentation**, not object detection: the model assigns every pixel to either road or background instead of drawing bounding boxes.

## Results

The current checkpoint was selected by validation IoU and evaluated on 624 held-out labeled images.

| Metric | Test score |
|---|---:|
| IoU / Jaccard | 0.4520 |
| Dice / F1 | 0.6225 |
| Precision | 0.6152 |
| Recall | 0.6301 |
| Pixel accuracy | 0.9670 |

Pixel accuracy is inflated by the large amount of background, so IoU and Dice are the primary quality indicators.

See [MODEL_CARD.md](MODEL_CARD.md) for training details, intended use, and limitations.

## Repository structure

```text
.
├── configs/
│   └── config.yaml
├── results/
│   └── test_metrics.json
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/imastikbaev/deepglobe-road-segmentation.git
cd deepglobe-road-segmentation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Dataset configuration

The dataset is intentionally not stored in Git.

Create a machine-specific configuration so your local path does not modify the tracked example:

```bash
cp configs/config.yaml configs/config.local.yaml
```

Set `dataset.path` in `configs/config.local.yaml` to either:

- the extracted DeepGlobe dataset directory; or
- the original dataset ZIP.

```yaml
dataset:
  path: "data/dataset_roads.zip"
```

Expected labeled pairs:

```text
100034_sat.jpg
100034_mask.png
```

Images without a matching mask are available for prediction but are excluded from training and evaluation.

## Trained checkpoint

The application expects:

```text
outputs/checkpoints/best_model.pth
```

You can produce it by training the model, or download `best_model.pth` from the latest GitHub Release and place it in that directory.

## Training

```bash
python src/train.py --config configs/config.local.yaml
```

Resume from the best checkpoint:

```bash
python src/train.py \
  --config configs/config.local.yaml \
  --resume \
  --learning-rate 0.0005
```

Training outputs are saved under `outputs/`:

```text
outputs/
├── checkpoints/best_model.pth
├── metrics.json
├── splits.json
├── training_history.json
└── training_log.csv
```

## Evaluation

```bash
python src/evaluate.py --config configs/config.local.yaml
```

## Prediction

Single image:

```bash
python src/predict.py \
  --config configs/config.local.yaml \
  --input path/to/image.jpg
```

Folder:

```bash
python src/predict.py \
  --config configs/config.local.yaml \
  --input path/to/images/
```

Predicted masks and comparison images are written to:

```text
outputs/predicted_masks/
outputs/visualizations/
```

## Development checks

```bash
pip install -r requirements-dev.txt
python -m compileall -q src
pytest
```

## Notes

- Inputs are resized to 256×256, which can remove fine road detail.
- Masks are resized with nearest-neighbor interpolation and binarized at 128.
- The split is deterministic with seed 42 and stored in `outputs/splits.json`.

## License

MIT. See [LICENSE](LICENSE).
