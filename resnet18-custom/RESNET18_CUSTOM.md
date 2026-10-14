# Custom ResNet18 Gesture Model

This is the third training pipeline. It combines the ResNet18 model and
preprocessing with the DINOv2 pipeline's private NEW-only session training.

## Important differences

- ResNet18 is created with `weights=None`.
- No HaGRID image or `ResNet18.pth` checkpoint is used.
- Training uses sessions `A-R` only.
- Validation uses sessions `S-V` only.
- Sessions `W-Z` remain reserved for final testing.
- Input preprocessing is shared with the ResNet18-HaGRID pipeline: 224x224
  letterboxing, the ResNet fill color, and ResNet normalization.

The expected private dataset layout is:

```text
Dataset/
├── gesture_1/
│   ├── session_A/
│   └── ...
├── gesture_2/
├── gesture_3/
└── no_gesture/
```

The class directory names are exactly `gesture_1`, `gesture_2`, `gesture_3`,
and `no_gesture`; each must contain at least one image in sessions `A-R`.

## Training

Install the pipeline requirements:

```bash
python3 -m pip install -r resnet18-custom/requirements.txt
```

Train from random initialization:

```bash
python3 resnet18-custom/scripts/train-resnet18-custom.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-custom
```

The training procedure follows DINOv2's NEW-only behavior: class-balanced
batches of 32, 281 batches per epoch, FP32 AdamW, three warmup epochs, cosine
decay, label smoothing, gradient clipping, and early stopping after four
non-improving validation epochs. The backbone and classifier learning rates
retain the DINOv2 defaults, while the complete ResNet18 backbone is trainable.

Resume an interrupted run with:

```bash
python3 resnet18-custom/scripts/train-resnet18-custom.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-custom \
  --resume /path/to/checkpoints/resnet18-custom/last.pt
```

The run writes `best.pt`, `last.pt`, `class_names.txt`, and
`training_config.json`. The checkpoints contain the model, optimizer, and
scheduler state and explicitly record random initialization and the session
split.

## Export

```bash
python3 resnet18-custom/scripts/export-resnet18-custom.py \
  --checkpoint /path/to/checkpoints/resnet18-custom/best.pt \
  --output-dir resnet18-custom/artifacts/export \
  --skip-tflite
```

The exported model returns four logits and a 512-dimensional ResNet embedding.
Preprocessing is not embedded; deployment must use the ResNet preprocessing.
