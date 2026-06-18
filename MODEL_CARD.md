# Model card

## Model

Four-level U-Net for binary road segmentation.

- Input: RGB satellite image
- Training size: 256×256
- Output: one-channel road logits
- Base channels: 32
- Loss: 0.5 BCE with logits + 0.5 Dice loss
- Prediction threshold: 0.5

## Training data

DeepGlobe Road Extraction Dataset.

The supplied archive contained 6,226 labeled image/mask pairs. A deterministic split with seed 42 produced:

| Split | Images |
|---|---:|
| Train | 4,980 |
| Validation | 622 |
| Test | 624 |

Unlabeled images were not included in training or evaluation.

## Evaluation

Best checkpoint: epoch 5, selected by validation IoU.

| Metric | Validation | Test |
|---|---:|---:|
| IoU | 0.4527 | 0.4520 |
| Dice | 0.6232 | 0.6225 |
| Precision | 0.6079 | 0.6152 |
| Recall | 0.6394 | 0.6301 |
| Pixel accuracy | 0.9682 | 0.9670 |

Exact test values are stored in [results/test_metrics.json](results/test_metrics.json).

## Intended use

- educational experiments with satellite road segmentation;
- comparing segmentation architectures or losses;
- visually testing road masks on imagery similar to DeepGlobe.

## Limitations

- This is an initial baseline, not a production mapping system.
- Resizing 1024×1024 imagery to 256×256 loses narrow roads and local detail.
- Predictions may confuse paths, clearings, rivers, shadows, and terrain boundaries with roads.
- Results can degrade on imagery from different sensors, regions, seasons, or resolutions.
- Output masks do not guarantee topological road connectivity.
- The model should not be used for safety-critical routing or autonomous navigation.
