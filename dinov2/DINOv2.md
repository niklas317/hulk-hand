# DINOv2 Gesture Model

## Current model

The current DINOv2 pipeline uses:

```text
DINOv2 ViT-S/14
Input: [batch, 3, 224, 224]
Embedding: 384 dimensions
Classifier: Linear(384, 4)
```

Transformer blocks `0-1` are frozen. Blocks `2-11`, the final LayerNorm, and
the classifier are trainable. Attention is replaced with a classical
Opset-13-compatible implementation.

The four output classes are:

```text
0: gesture_1  (one)
1: gesture_2  (two)
2: gesture_3  (stop)
3: no_gesture
```

## Dataset

The active pipeline uses only the custom NEW dataset. OLD/HaGRID samples are
not loaded.

```text
A-R: training
S-V: validation
W-Z: final test
```

W-Z must remain untouched until model development and model selection are
complete.

The expected dataset layout is:

```text
Dataset/
├── gesture_1/
│   ├── session_A/
│   └── ...
├── gesture_2/
├── gesture_3/
└── no_gesture/
```

## V12 result

The current best checkpoint is the V12 development checkpoint:

```text
Best epoch:          29
NEW train accuracy:  98.61%
NEW validation:      89.00%
Generalization gap:  9.61 percentage points
```

Validation by session:

```text
S: 94.20%
T: 82.00%
U: 83.60%
V: 96.20%
```

The remaining weakness is `gesture_2`, particularly in sessions T and U.
The final W-Z test has not been opened.

## Requirements

```bash
python -m pip install -r dinov2/requirements.txt
```

DINOv2 is loaded through `torch.hub`. The upstream repository and pretrained
weights retain their own licenses and terms.

## Training

```bash
python dinov2/scripts/train-dinov2.py \
    --data-dir /path/to/Dataset \
    --output-dir /path/to/checkpoints/dinov2 \
    --horizontal-flip
```

Resume an interrupted run:

```bash
python dinov2/scripts/train-dinov2.py \
    --data-dir /path/to/Dataset \
    --output-dir /path/to/checkpoints/dinov2 \
    --resume /path/to/checkpoints/dinov2/last.pt \
    --horizontal-flip
```

Training uses FP32, AdamW, class-balanced NEW-only batches, 32 samples per
batch, 281 batches per epoch, cosine decay, and early stopping patience 4.

## Diagnosis

Diagnosis evaluates NEW training sessions A-R and validation sessions S-V.
It does not evaluate W-Z.

```bash
python dinov2/scripts/diagnose-dinov2.py \
    --data-dir /path/to/Dataset \
    --checkpoint /path/to/checkpoints/dinov2/best.pt
```

## Final test evaluation

The final test script is intentionally reserved for one evaluation on W-Z:

```bash
python dinov2/scripts/evaluate-dinov2.py \
    --data-dir /path/to/Dataset \
    --checkpoint /path/to/checkpoints/dinov2/best.pt
```

Do not run this during model development or checkpoint selection.

## ONNX export

The exporter produces fixed-shape Opset-13 ONNX files and validates the graph
with ONNX Runtime before finalizing the output.

```bash
python dinov2/scripts/export-dinov2.py \
    --checkpoint /path/to/checkpoints/dinov2/best.pt \
    --output /path/to/hulk_hand_dinov2.onnx \
    --gelu-approximation \
    --rewrite-pow
```

The target interface is:

```text
Input:     [1, 3, 224, 224]
Logits:    [1, 4]
Embedding: [1, 384]
Opset:     13
```

Export-time compatibility changes do not modify the checkpoint or require
retraining. Resize conversion is not recommended because changing DINOv2's
positional-embedding interpolation caused a significant numerical deviation.

## Preprocessing

Stored images are RGB JPEGs. Inference uses 224x224 images with ImageNet
normalization:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Preprocessing is not embedded in the ONNX graph and must be reproduced by the
deployment application.
