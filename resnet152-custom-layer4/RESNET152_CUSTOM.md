# Custom ResNet152 Layer4 Gesture Model

This pipeline mirrors the private-session ResNet pipeline but uses a ResNet152 backbone and
trains from random initialization and does not load HaGRID or any other
starting checkpoint.

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
python3 -m pip install -r resnet152-custom-layer4/requirements.txt

python3 resnet152-custom-layer4/scripts/train-resnet152-layer4.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet152-custom-layer4
```

Resume an interrupted run:

```bash
python3 resnet152-custom-layer4/scripts/train-resnet152-layer4.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet152-custom-layer4 \
  --resume /path/to/checkpoints/resnet152-custom-layer4/last.pt
```

## Outputs

The PyTorch model and exported ONNX model both provide:

```text
logits:    [batch, 4]
embedding: [batch, 2048]
```

The embedding is the penultimate ResNet152 feature vector and remains the
second output, matching the dual-output ResNet18 export style.

## Export

```bash
python3 resnet152-custom-layer4/scripts/export-resnet152-layer4.py \
  --checkpoint /path/to/checkpoints/resnet152-custom-layer4/best.pt \
  --output-dir resnet152-custom-layer4/artifacts/export \
  --skip-tflite
```

Verify the exported graph:

```bash
python3 resnet152-custom-layer4/scripts/check-onnx-opset.py \
  resnet152-custom-layer4/artifacts/export/ResNet152_layer4_4class_opset13_dual_output.onnx \
  --expect 13
```
