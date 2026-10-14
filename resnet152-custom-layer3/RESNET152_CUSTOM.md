# Custom ResNet152 Layer3 Gesture Model

This pipeline mirrors the custom ResNet training pipeline but exposes the
pooled `layer3` feature as output two. It trains from random initialization and
does not load HaGRID or any other starting checkpoint.

## Dataset

```text
Dataset/
├── gesture_1/
│   ├── session_A/
│   └── ...
├── gesture_2/
├── gesture_3/
└── no_gesture/
```

Sessions `A-R` are used for training, `S-V` for validation, and `W-Z` remain
reserved for final testing. Other directories are ignored.

## Training

The pipeline uses the same ResNet preprocessing and DINO-style training as the
custom ResNet18 pipeline: 224x224 letterboxing, balanced batches, FP32 AdamW,
warmup, cosine decay, label smoothing, gradient clipping, early stopping, and
`tqdm` progress bars.

```bash
python3 -m pip install -r resnet152-custom-layer3/requirements.txt

python3 resnet152-custom-layer3/scripts/train-resnet152-layer3.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet152-custom-layer3
```

Resume an interrupted run:

```bash
python3 resnet152-custom-layer3/scripts/train-resnet152-layer3.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet152-custom-layer3 \
  --resume /path/to/checkpoints/resnet152-custom-layer3/last.pt
```

## Outputs

The PyTorch model and exported ONNX model both provide:

```text
logits:    [batch, 4]
embedding: [batch, 1024]
```

The embedding is the global-average-pooled output of `layer3`. Classification
still continues through `layer4`, so the full ResNet152 remains trainable.

## Export

```bash
python3 resnet152-custom-layer3/scripts/export-resnet152-layer3.py \
  --checkpoint /path/to/checkpoints/resnet152-custom-layer3/best.pt \
  --output-dir resnet152-custom-layer3/artifacts/export \
  --skip-tflite
```

Verify the exported graph:

```bash
python3 resnet152-custom-layer3/scripts/check-onnx-opset.py \
  resnet152-custom-layer3/artifacts/export/ResNet152_layer3_4class_opset13_dual_output.onnx \
  --expect 13
```
