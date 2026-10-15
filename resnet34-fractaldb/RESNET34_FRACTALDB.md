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

The reference trainer uses session-balanced class batches, warmup/cosine
scheduling, and staged fine-tuning while keeping `layer1` frozen.

## Training

```bash
python3 -m pip install -r resnet34-fractaldb/requirements.txt

python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet34-fractaldb
```

The first run downloads `model.safetensors` from Hugging Face. The checkpoint
is cached locally by `huggingface_hub` and is not stored in this repository.

Resume an interrupted run:

```bash
python3 resnet34-fractaldb/scripts/train-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet34-fractaldb \
  --resume /path/to/checkpoints/resnet34-fractaldb/last.pt
```

## Diagnosis

Evaluate the validation sessions `S-V`:

```bash
python3 resnet34-fractaldb/scripts/diagnose-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet34-fractaldb/best.pt \
  --split validation
```

The report includes overall, per-class, and per-session accuracy plus a
confusion matrix. After model selection, evaluate the untouched test sessions
`W-Z` once with:

```bash
python3 resnet34-fractaldb/scripts/diagnose-resnet34-fractaldb.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet34-fractaldb/best.pt \
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
  --checkpoint /path/to/checkpoints/resnet34-fractaldb/best.pt \
  --output-dir resnet34-fractaldb/artifacts/export \
  --skip-tflite
```

Verify the exported graph:

```bash
python3 resnet34-fractaldb/scripts/check-onnx-opset.py \
  resnet34-fractaldb/artifacts/export/ResNet34_4class_opset13_dual_output.onnx \
  --expect 13
```

## DINOv2 knowledge distillation

The standalone distillation trainer uses a local, already fine-tuned DINOv2
checkpoint as teacher. It validates the teacher on `S-V` before any student
training and refuses to continue if the teacher is below the configured
threshold (default: 80%). The teacher is never stored in the student
checkpoint.

Run the required preflight smoke test first:

```bash
python3 resnet34-fractaldb/scripts/train-distill-dinov2-resnet34.py \
  --teacher-checkpoint /path/to/dinov2/best.pt \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet34-distill-smoke \
  --smoke-test
```

Start a complete run in a fresh output directory:

```bash
python3 resnet34-fractaldb/scripts/train-distill-dinov2-resnet34.py \
  --teacher-checkpoint /path/to/dinov2/best.pt \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet34-distill
```

This is the E5-v2 configuration: CE weight `0.7`, KD weight `0.3`, temperature
`2.0`, label smoothing `0.05`, and a maximum of 35 epochs. The student stages
are classifier-only (epoch 1), `layer4+classifier` (epochs 2-3), and
`layer2+layer3+layer4+stem+classifier` from epoch 4. Frozen BatchNorm modules
remain in evaluation mode; newly trainable modules use regular BatchNorm.
Before training, the script reports teacher accuracy on clean and augmented
`A-R` plus clean `S-V` data and stops if augmentation degrades the teacher
beyond the configured limit (`--max-teacher-augmentation-drop`, default 10%).
Teacher and student receive the same augmented image; only their normalization
paths are separate. The output contains only the ResNet34 student weights.
