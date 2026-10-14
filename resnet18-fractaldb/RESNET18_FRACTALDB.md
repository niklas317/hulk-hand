# FractalDB ResNet18 Gesture Model

This pipeline fine-tunes the standard torchvision ResNet18 initialized from
the Hugging Face FractalDB checkpoint:

```text
totti0223/resnet18_fractaldb_10000
```

The original 10,000-class FractalDB classifier is discarded and replaced with
the four gesture classes. Only the first residual stage, `layer1`, is frozen.
The stem (`conv1` and `bn1`), `layer2`, `layer3`, `layer4`, and the new
classifier remain trainable so the backbone can adapt to the exact
ResNet-HaGRID preprocessing.

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

This pipeline uses the exact preprocessing from `resnet18-hagrid`:

- RGB conversion
- Aspect-ratio-preserving 224x224 letterboxing
- ResNet-HaGRID fill color
- ResNet-HaGRID normalization
- Existing ResNet training augmentations

## Training

```bash
python3 -m pip install -r resnet18-fractaldb/requirements.txt

python3 resnet18-fractaldb/scripts/train-resnet18-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-fractaldb
```

The first run downloads `model.safetensors` from Hugging Face. The checkpoint
is cached locally by `huggingface_hub` and is not stored in this repository.

Resume an interrupted run:

```bash
python3 resnet18-fractaldb/scripts/train-resnet18-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-fractaldb \
  --resume /path/to/checkpoints/resnet18-fractaldb/last.pt
```

## Diagnosis

Evaluate the validation sessions `S-V`:

```bash
python3 resnet18-fractaldb/scripts/diagnose-resnet18-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet18-fractaldb/best.pt \
  --split validation
```

The report includes overall, per-class, and per-session accuracy plus a
confusion matrix. After model selection, evaluate the untouched test sessions
`W-Z` once with:

```bash
python3 resnet18-fractaldb/scripts/diagnose-resnet18-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet18-fractaldb/best.pt \
  --split test
```

## Outputs

The model returns:

```text
logits:    [batch, 4]
embedding: [batch, 512]
```

The embedding is the standard pooled ResNet18 `layer4` feature and remains the
second output.

## Export

```bash
python3 resnet18-fractaldb/scripts/export-resnet18-fractaldb.py \
  --checkpoint /path/to/checkpoints/resnet18-fractaldb/best.pt \
  --output-dir resnet18-fractaldb/artifacts/export \
  --skip-tflite
```

Verify the exported graph:

```bash
python3 resnet18-fractaldb/scripts/check-onnx-opset.py \
  resnet18-fractaldb/artifacts/export/ResNet18_4class_opset13_dual_output.onnx \
  --expect 13
```
