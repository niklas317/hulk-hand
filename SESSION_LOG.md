# Session Log

## 2026-08-27 - V9

- Backbone: DINOv2 ViT-S/14
- Training setup: last 8 transformer blocks fine-tuned, layerwise LR decay
- Outcome:
  - Epoch 21/40
  - Train loss: 0.3193
  - Train accuracy: 94.70%
  - OLD val loss: 0.3496
  - OLD val accuracy: 93.46%
  - NEW val loss: 0.5539
  - NEW val accuracy: 84.80%
  - Best NEW validation accuracy: 85.80%
  - Early stopping triggered at 8/8
- Checkpoints: to be added manually later

## 2026-08-27 - V10

- Dataset: enhanced V10 dataset
- Backbone: DINOv2 ViT-S/14
- Training setup: same as V9, last 8 transformer blocks fine-tuned with layerwise LR decay
- Preprocessing: blur threshold 222, contrast threshold 35, other thresholds default
- Training length: 30 epochs
- Outcome:
  - Train loss: 0.2819
  - Train accuracy: 96.90%
  - OLD val loss: 0.3348
  - OLD val accuracy: 94.23%
  - NEW val loss: 0.5248
  - NEW val accuracy: 86.87%
  - LR backbone: 0.00000032..0.00000101
  - LR classifier: 0.00004054
  - Early stopping triggered at 8/8
- Checkpoints: to be added manually later

## 2026-08-27 - V11

- Dataset: enhanced V10 dataset
- Backbone: DINOv2 ViT-S/14
- Training setup: last 10 transformer blocks fine-tuned, layerwise LR decay
- Preprocessing: blur threshold 222, contrast threshold 35, other thresholds default
- Training length: 30 epochs
- Early stopping: patience 4
- Outcome:
  - Epoch 27/40
  - Train loss: 0.3068
  - Train accuracy: 96.01%
  - OLD val loss: 0.3612
  - OLD val accuracy: 92.85%
  - NEW val loss: 0.5235
  - NEW val accuracy: 86.67%
  - LR backbone: 0.00000018..0.00000078
  - LR classifier: 0.00006271
  - Early stopping triggered at 4/4
- Checkpoints: to be added manually later

## V12 - NEW-only training

- Dataset: NEW custom recordings only
- OLD/HaGRID samples: not loaded
- Backbone: DINOv2 ViT-S/14
- Training setup: last 10 transformer blocks fine-tuned with layerwise LR decay
- Training data: sessions A-R
- Validation data: sessions S-V
- Test data: sessions W-Z remain untouched
- Batch sampling: class-balanced, 32 samples per batch, 8 per class
- Checkpoint epoch: 29
- Outcome:
  - NEW train accuracy: 98.61%
  - NEW validation accuracy: 89.00%
  - Generalization gap: 9.61 percentage points
  - Saved/recomputed validation difference: +0.0000 pp
- Validation by session:
  - S: 94.20%
  - T: 82.00%
  - U: 83.60%
  - V: 96.20%
- Main remaining error: gesture_2, especially sessions T and U
- Checkpoints: `checkpoints/training/v12/best.pt` and `last.pt`
