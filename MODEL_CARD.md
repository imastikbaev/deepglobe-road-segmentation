# Model card

## Model

Four-level U-Net for binary road segmentation.

- Architecture: U-Net
- Input: RGB aerial or satellite image
- Input size: `256×256`
- Output: one-channel road logits
- Base channels: `32`
- Loss: `0.5 × BCEWithLogits + 0.5 × Dice`
- Prediction threshold: `0.5`
- Release tag: [`v1.0-uavid`](https://github.com/imastikbaev/deepglobe-road-segmentation/releases/tag/v1.0-uavid)
- Checkpoint: `best_model_uavid_finetuned_iou7168.pth`
- SHA-256: `32fd1735b805f27bf31048f14a276ff3c3d5cba78b7ca2f63c6d49ea6dea1468`

## Training data

The model was pretrained on the DeepGlobe Road Extraction Dataset and then
fine-tuned on UAVid.

UAVid semantic labels were converted to binary masks by treating the UAVid road
color (`RGB 128, 64, 128`) as foreground and all other classes as background.
The prepared fine-tuning dataset contained 670 image/mask pairs.

A deterministic 80/10/10 split with seed 42 produced:

| Split | Images |
|---|---:|
| Train | 536 |
| Validation | 67 |
| Test | 67 |

## Fine-tuning configuration

- Starting point: DeepGlobe U-Net checkpoint
- Fine-tuning epochs: through epoch 48
- Best checkpoint: epoch 47
- Learning rate: `0.00005`
- Batch size: `2`
- Optimizer: AdamW
- Weight decay: `0.0001`
- Augmentation: horizontal/vertical flips, rotation, brightness, and contrast
- Selection metric: validation IoU

The tracked configuration is
[`configs/config.uavid_finetune.yaml`](configs/config.uavid_finetune.yaml).

## Evaluation

The checkpoint from epoch 47 was selected with validation IoU `0.6998`.

| Metric | Validation | Test |
|---|---:|---:|
| IoU | 0.6998 | **0.7168** |
| Dice | 0.8234 | **0.8350** |
| Precision | 0.7896 | **0.8149** |
| Recall | 0.8602 | **0.8561** |
| Pixel accuracy | 0.9588 | **0.9550** |
| BCE + Dice loss | 0.2051 | **0.1963** |

Validation values were read from the released checkpoint; test values come
from the held-out test evaluation. Machine-readable results are stored in
[`results/test_metrics.json`](results/test_metrics.json).

## Intended use

- road-mask extraction from aerial and satellite imagery;
- educational semantic-segmentation experiments;
- transfer-learning and domain-adaptation comparisons;
- visual inspection or preprocessing for non-safety-critical GIS workflows.

## Limitations

- The evaluation set contains 67 images, so results should be interpreted with
  the test-set size in mind.
- Resizing imagery to `256×256` removes fine spatial detail.
- Predictions may confuse paths, clearings, roofs, shadows, rivers, or terrain
  boundaries with roads.
- Performance can degrade on imagery from different sensors, regions, seasons,
  resolutions, and flight altitudes.
- Output masks do not guarantee topological road connectivity.
- The model must not be used as the sole input for safety-critical routing,
  autonomous navigation, or emergency-response decisions.
