# ResNet18 ImageNet-1K Pipeline

This is an independent technical experiment using the unchanged torchvision
ResNet18 architecture initialized with `ResNet18_Weights.IMAGENET1K_V1`. The
classifier is replaced with four outputs in the fixed order:

```text
gesture_1, gesture_2, gesture_3, no_gesture
```

This pipeline uses no hand detector, crop, MediaPipe, attention module, or
knowledge distillation. The model receives only four-class logits and is
intended to test whether natural ImageNet-1K features generalize better than
the completed FractalDB experiments.

## Installation

```bash
python3 -m pip install -r resnet18-in1k/requirements.txt
```

## Dataset

The existing offline-prepared 224x224 dataset is used directly:

```text
Dataset/
├── gesture_1/session_A ... session_Z/
├── gesture_2/session_A ... session_Z/
├── gesture_3/session_A ... session_Z/
└── no_gesture/session_A ... session_Z/
```

Training uses A-R, validation uses S-V, and W-Z is reserved for the final
test. The balanced batch sampler emits equal class counts and cycles through
training sessions. No validation or test image is used for optimization.

## Preprocessing and augmentation

Stored images are converted to RGB, resized to the fixed 224x224 input, and
normalized with ImageNet values:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Training uses stronger but still non-cropping ColorJitter and one bilinear
affine transform: rotation up to 12 degrees, translation up to 6%, scale
0.94-1.06, black fill. This is intended to reduce camera/background and burst
session overfitting without removing fingers.
There is no crop, blur, or RandomErasing. Validation, test, and deployment use
the deterministic evaluation transform without augmentation.

## Training

Use a fresh output directory for a new run:

```bash
python3 resnet18-in1k/scripts/train.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-in1k
```

The current robust-training configuration is batch size 32, at most 30 epochs,
AdamW, weight decay `5e-4`, label smoothing `0.05`, gradient clipping `1.0`,
warmup plus cosine decay, and early-stopping patience 7. Epochs 1-4 train
only `fc`; from epoch 5 the complete backbone is trainable with layer-specific
learning rates. BatchNorm statistics are frozen by default to reduce
session-specific drift. The rates are configurable in `scripts/train.py`:

```text
conv1+bn1  1e-5     layer1 1e-5     layer2 2e-5
layer3    3e-5     layer4 5e-5     fc     3e-4
```

The weighted training loss uses class weights `(1.15, 1.30, 1.15, 0.70)` for
`gesture_1`, `gesture_2`, `gesture_3`, and `no_gesture`, respectively. This
directly targets the observed gesture-to-`no_gesture` error pattern. Disable
BatchNorm freezing only for a controlled comparison with `--no-freeze-bn`.

Resume a run with matching configuration:

```bash
python3 resnet18-in1k/scripts/train.py \
  --data-dir /path/to/private/Dataset \
  --output-dir /path/to/checkpoints/resnet18-in1k \
  --resume /path/to/checkpoints/resnet18-in1k/last.pt
```

Outputs include `best.pt`, `last.pt`, `metrics.json`, and
`training_config.json`. `best.pt` is selected by validation accuracy and also
stores the corresponding balanced accuracy.

## Evaluation

Before changing the training recipe, inspect the session/domain failure mode
with deterministic validation and layer4 Grad-CAM:

```bash
python3 resnet18-in1k/scripts/diagnose.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet18-in1k-robust-v2/best.pt \
  --output-dir /path/to/checkpoints/resnet18-in1k-robust-v2/diagnosis
```

This writes validation logits, per-class/per-session metrics, input RGB
statistics by session, and Grad-CAM grids for up to five errors per class and
session. It does not train or modify the checkpoint.

Validation diagnosis:

```bash
python3 resnet18-in1k/scripts/evaluate.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet18-in1k/best.pt \
  --split validation
```

Final test evaluation after model selection:

```bash
python3 resnet18-in1k/scripts/evaluate.py \
  --data-dir /path/to/private/Dataset \
  --checkpoint /path/to/checkpoints/resnet18-in1k/best.pt \
  --split test
```

## ONNX export

Export a fixed-shape, logits-only Opset-13 model:

```bash
python3 resnet18-in1k/scripts/export.py \
  --checkpoint /path/to/checkpoints/resnet18-in1k/best.pt \
  --output /path/to/resnet18-in1k.onnx \
  --compare
```

The exported interface is input `[1,3,224,224]` and output `[1,4]`. The
optional comparison checks one deterministic random input with ONNX Runtime.
The ONNX model does not include preprocessing; deployment must reproduce RGB
conversion, the fixed 224x224 resize, and ImageNet normalization.

## License and pretrained weights

The experiment uses the torchvision ImageNet-1K pretrained checkpoint
`ResNet18_Weights.IMAGENET1K_V1`. The license of the torchvision implementation
must not be equated automatically with the usage rights of the pretrained
weights or the original ImageNet dataset. Any commercial release requires a
separate rights review. This directory is initially for technical evaluation
only.
