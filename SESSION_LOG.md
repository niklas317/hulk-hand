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
