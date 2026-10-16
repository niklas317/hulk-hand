# DINOv2 Gesture Model

## Current model

The current DINOv2 pipeline uses:

```text
DINOv2 ViT-S/14
Input: [batch, 3, 224, 224]
Embedding: 384 dimensions
Classifier: Linear(384, 4)
```

Transformer blocks `0-1` are frozen. Blocks `2-11`, the final LayerNorm, and the classifier are trainable. Attention is replaced with a classical Opset-13-compatible implementation.

The four output classes are:

```text
0: gesture_1  (one)
1: gesture_2  (two)
2: gesture_3  (stop)
3: no_gesture
```

## Dataset

The active pipeline uses a private dataset that was recorded. It features 26 different recording sessions with different scenes and compositions, but all feature a person with a green glove on one hand showing the gesture.

```text
A-R: training
S-V: validation
W-Z: final test
```

W-Z must remain untouched until model development and model selection are complete.

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

## Result

The current best checkpoint:

```text
Train accuracy:  98.61%
Validation accuracy:      89.00%
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

## Requirements

```bash
python -m pip install -r dinov2/requirements.txt
```

DINOv2 is loaded through `torch.hub`. The upstream repository and pretrained
weights retain their own licenses and terms.

## Offline Dataset Preprocessing

The images used for the training run were preprocessed **offline**, before being passed to the training script. The preprocessing step letterboxes each original image to **224 × 224** using the `fit-pad` mode: it preserves the original aspect ratio, centers the image, fills unused space with **black padding**, and saves the result as an RGB JPEG (default quality: 90).

Additionally, an ImageNet normalization is applied after te resize during runtime:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

To prepare a dataset with the same class and session folder structure, run:

```bash
python dinov2/preprocessing/preprocess_dinov2_images.py \
    --input-dir /path/to/raw/Dataset \
    --output-dir /path/to/Dataset \
    --mode fit-pad \
    --image-size 224 \
    --quality 90
```

The input and output directories must be separate, and the output directory must not be inside the input directory. The script retains the relative class/session folder structure.

**Train on the preprocessed dataset**, as shown in the Training section below. The dataset loader in `dinov2/scripts/dataset.py` applies `Resize((224, 224))` when reading images; because the stored images are already 224 × 224 squares, this does not introduce aspect-ratio distortion. Training then applies its random data augmentations and ImageNet normalization. Validation and testing use the same stored images without random augmentation. 

During webcam inference, `dinov2/preprocessing/dinov2_preprocessing.py` performs the equivalent **fit-pad letterboxing on each incoming frame** before normalization. The ONNX model does not include preprocessing, so deployment applications must reproduce this step. Unlike offline dataset preparation, webcam inference does not save a JPEG intermediate.

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

Training uses FP32, AdamW, class-balanced NEW-only batches, 32 samples per batch, 281 batches per epoch, cosine decay, and early stopping patience 4.

## Diagnosis

Diagnosis evaluates NEW training sessions A-R and validation sessions S-V. It does not evaluate W-Z.

```bash
python dinov2/scripts/diagnose-dinov2.py \
    --data-dir /path/to/Dataset \
    --checkpoint /path/to/checkpoints/dinov2/best.pt
```

## Final test evaluation

Following model selection, the finalized checkpoint was evaluated on held-out sessions W-Z with:

```bash
python dinov2/scripts/evaluate-dinov2.py \
    --data-dir /path/to/Dataset \
    --checkpoint /path/to/checkpoints/dinov2/best.pt
```

Final held-out test results (2,000 images):

```text
Overall accuracy:    89.35% (1,787 / 2,000)

Per-class accuracy:
gesture_1:           74.75% (299 / 400)
gesture_2:           95.50% (382 / 400)
gesture_3:           99.50% (398 / 400)
no_gesture:          88.50% (708 / 800)
```

Confusion matrix (rows = true class, columns = predicted class):

| Actual / Predicted | `gesture_1` | `gesture_2` | `gesture_3` | `no_gesture` |
|---|---:|---:|---:|---:|
| `gesture_1` | 299 | 9 | 1 | 91 |
| `gesture_2` | 3 | 382 | 0 | 15 |
| `gesture_3` | 0 | 0 | 398 | 2 |
| `no_gesture` | 25 | 41 | 26 | 708 |

The largest test-set error category is `gesture_1` predicted as `no_gesture` (91 of 400 `gesture_1` images). The held-out test accuracy is close to the 89.00% validation accuracy, but per-class performance differs. These results were obtained after finalization and should not be used to reselect or tune the checkpoint.

## ONNX export

The exporter produces fixed-shape Opset-13 ONNX files and validates the graph with ONNX Runtime before finalizing the output.

```bash
python dinov2/scripts/export-dinov2.py \
    --checkpoint /path/to/checkpoints/dinov2/best.pt \
    --output /path/to/hulk_hand_dinov2.onnx \
    --attention-variant original \
    --gelu-approximation \
    --rewrite-pow
```

The target interface is:

```text
Input:     [1, 3, 224, 224]
Outputs:   logits [1, 4], embedding [1, 384]
Opset:     13
```

The exporter supports four attention implementations. They all reuse the
trained DINOv2 weights and are mathematically equivalent in PyTorch inference:

```text
original       existing explicit attention (5-D QKV reshape, 4-D MatMul)
split-qkv      split Q/K/V before head reshaping (no 5-D combined QKV tensor)
flatten-heads  flatten batch and heads for 3-D attention MatMul
separate-heads unroll the six heads and use only 2-D attention MatMul
```

Use a separate output file for each variant. The exporter validates Opset 13,
the fixed input/output shapes, ONNX graph integrity, and the maximum absolute
PyTorch-vs-ONNX differences for both logits and embeddings. It also reports
inferred attention tensor ranks and operator counts. Converter compatibility
and full validation remain external tests and are not claimed by this
source-level implementation.

Export-time compatibility changes do not modify the checkpoint or require retraining. Resize conversion is not recommended because changing DINOv2's positional-embedding interpolation caused a significant numerical deviation.
