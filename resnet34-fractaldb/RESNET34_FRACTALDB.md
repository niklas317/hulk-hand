# FractalDB ResNet34 Gesture Model

This pipeline fine-tunes the standard torchvision ResNet34 initialized from
the Hugging Face FractalDB checkpoint:

```text
totti0223/resnet34_fractaldb_10000
```

The original 10,000-class FractalDB classifier is discarded and replaced with
the four gesture classes. Only the first residual stage, `layer1`, is frozen.
The stem (`conv1` and `bn1`), `layer2`, `layer3`, `layer4`, and the new
classifier remain trainable so the backbone can adapt to the private dataset.

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

## Preprocessing

This pipeline uses FractalDB-compatible ImageNet normalization with
geometry-preserving preprocessing rather than the HaGRID normalization:

- RGB conversion
- Aspect-ratio-preserving letterbox to 224x224 with the ResNet fill color
- ImageNet normalization: mean `(0.485, 0.456, 0.406)`
- ImageNet normalization: std `(0.229, 0.224, 0.225)`
- ColorJitter plus one conservative bilinear affine transform
- RandomErasing disabled

The `resnet18-hagrid` preprocessing remains unchanged.

Training uses session-balanced class batches and three explicit experiments:

- `E1`: corrected augmentation and the previous staged training settings.
- `E2`: backbone LR `1e-5`, classifier LR `3e-4`, label smoothing `0.05`,
  patience `7`; classifier-only for epochs 1-3, then `layer4+classifier`.
- `E3`: E2 plus `--freeze-bn` for all `BatchNorm2d` modules.

All experiments keep `layer1` frozen. Each run writes an augmentation preview
before training; inspect it for finger cropping before continuing.

## Training

```bash
python3 -m pip install -r resnet34-fractaldb/requirements.txt

python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --experiment E1 \
  --output-dir /path/to/checkpoints/resnet34-fractaldb-e1
```

Run E2 or E3 in separate directories:

```bash
python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --experiment E2 \
  --output-dir /path/to/checkpoints/resnet34-fractaldb-e2

python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --experiment E3 \
  --output-dir /path/to/checkpoints/resnet34-fractaldb-e3 \
  --freeze-bn
```

The first run downloads `model.safetensors` from Hugging Face. The checkpoint
is cached locally by `huggingface_hub` and is not stored in this repository.

Resume an interrupted run:

```bash
python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --experiment E2 \
  --output-dir /path/to/checkpoints/resnet34-fractaldb-e2 \
  --resume /path/to/checkpoints/resnet34-fractaldb-e2/last.pt
```

## Diagnosis

Evaluate the validation sessions `S-V`:

```bash
python3 resnet34-fractaldb/scripts/diagnose-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet34-fractaldb-e2/best.pt \
  --split validation
```

The report includes overall, per-class, and per-session accuracy plus a
confusion matrix. After model selection, evaluate the untouched test sessions
`W-Z` once with:

```bash
python3 resnet34-fractaldb/scripts/diagnose-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet34-fractaldb-e2/best.pt \
  --split test
```

## Outputs

The model returns:

```text
logits:    [batch, 4]
embedding: [batch, 512]
```

The embedding is the standard pooled ResNet34 `layer4` feature and remains the
second output.

## Export

```bash
python3 resnet34-fractaldb/scripts/export-resnet34-fractaldb.py \
  --checkpoint /path/to/checkpoints/resnet34-fractaldb-e2/best.pt \
  --output-dir resnet34-fractaldb-e2/export \
  --skip-tflite
```

Verify the exported graph:

```bash
python3 resnet34-fractaldb/scripts/check-onnx-opset.py \
  resnet34-fractaldb/artifacts/export/ResNet34_4class_opset13_dual_output.onnx \
  --expect 13
```
