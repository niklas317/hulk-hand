# hulk-hand

`hulk-hand` is a lightweight static hand gesture classification project based on a pretrained **HaGRID ResNet-18**.

The model is fine-tuned to recognize a reduced set of hand gestures performed with a **colored glove** in indoor camera scenes.

The project covers the complete model creation pipeline:

- image capture
- image preprocessing
- dataset organization
- OLD / NEW dataset balancing
- fine-tuning a pretrained HaGRID ResNet-18
- validation and checkpointing
- diagnostic evaluation
- final test evaluation
- export to ONNX
- explicit verification of **ONNX Opset 13**
- numerical comparison between PyTorch and ONNX Runtime

Deployment and downstream inference applications are outside the scope of this repository.

---

# Quick command reference

All commands below assume they are executed from the repository root:

```bash
cd /path/to/hulk-hand
```

Every CLI script can show its available arguments with:

```bash
python src/<script>.py --help
```

For example:

```bash
python src/train.py --help
```

## 1. Inspect the dataset

```bash
python src/dataset.py \
    --data-dir /path/to/Dataset
```

## 2. Check the pretrained HaGRID backbone

```bash
python src/model.py \
    --checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --num-classes 4
```

## 3. Run the complete FP32 smoke test

```bash
python src/smoke_test.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth
```

A successful test ends with:

```text
SMOKE TEST PASSED
```

## 4. Start a fresh training experiment

Example for experiment `v3`:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --output-dir checkpoints/training/v3
```

Never reuse the output directory of an older independent experiment.

For example:

```text
checkpoints/training/
├── v1/
│   ├── best.pt
│   └── last.pt
├── v2/
│   ├── best.pt
│   └── last.pt
└── v3/
    ├── best.pt
    └── last.pt
```

## 5. Resume an interrupted experiment

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v3 \
    --resume checkpoints/training/v3/last.pt
```

Do **not** use `--hagrid-checkpoint` when resuming an existing fine-tuning run.

## 6. Diagnose the best checkpoint

```bash
python src/diagnose.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v3/best.pt
```

This evaluates:

```text
NEW train A-R
NEW validation S-V
validation sessions S, T, U and V individually
```

It does **not** touch the final test sessions W-Z.

## 7. Final test evaluation

Only run this after model development and model selection are finished:

```bash
python src/evaluate.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v3/best.pt
```

This evaluates only:

```text
W-Z
```

## 8. Export the final model to ONNX

```bash
python src/export_onnx.py \
    --checkpoint checkpoints/training/v3/best.pt \
    --output models/hulk-hand.onnx
```

---

# Architecture

The project starts from the pretrained HaGRID ResNet-18 backbone.

```text
Input
[batch, 3, 224, 224]
        │
        ▼
      conv1
        │
       bn1
        │
      layer1
        │
      layer2
        │
      layer3
        │
        └──────────── FROZEN
        │
        ▼
      layer4
        │
        └──────────── TRAINABLE
        │
        ▼
 AdaptiveAvgPool
        │
        ▼
 512-D embedding
        │
        ├────────────► embedding output
        │
        ▼
   classifier
    512 → N
        │
        ▼
      logits
```

The following parts are frozen during fine-tuning:

```text
conv1
bn1
layer1
layer2
layer3
```

The following parts remain trainable:

```text
layer4
classifier
```

The BatchNorm statistics of the frozen backbone are also kept frozen.

The original HaGRID `leading_hand` head is not used.

The number of output classes is determined automatically from the dataset.

---

# Model outputs

The PyTorch model returns:

```python
{
    "logits": logits,
    "embedding": embedding,
}
```

with:

```text
logits:
[batch_size, num_classes]

embedding:
[batch_size, 512]
```

The 512-dimensional embedding is the feature vector directly after the final average pooling layer and before the classifier.

---

# Repository structure

```text
hulk-hand/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── capture_burst.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── sampler.py
│   ├── model.py
│   ├── smoke_test.py
│   ├── train.py
│   ├── diagnose.py
│   ├── evaluate.py
│   └── export_onnx.py
│
├── checkpoints/
│   ├── pretrained/
│   │   └── hagrid_resnet18.pth
│   │
│   └── training/
│       ├── v1/
│       ├── v2/
│       └── ...
│
└── models/
    └── hulk-hand.onnx
```

The dataset itself may be stored outside the repository.

All relevant paths are supplied through command-line arguments.

---

# Image capture

`src/capture_burst.py` captures a fixed number of images from a V4L2 camera.

Example:

```bash
python src/capture_burst.py \
    --device /dev/video0 \
    --output-dir data/raw/gesture_1/session_A \
    --fps 5 \
    --num-images 100
```

The preview window is displayed during recording.

The preview is horizontally flipped for display only. The images written to disk retain the original camera frame.

The default delay before recording starts is:

```text
1 second
```

It can be changed with:

```bash
--delay 2
```

The capture can be aborted with:

```text
q
```

## Capture flags

| Flag | Meaning |
|---|---|
| `--device` | V4L2 source such as `/dev/video0` |
| `--output-dir` | destination directory |
| `--fps` | desired capture rate |
| `--num-images` | number of frames to store |
| `--delay` | delay before burst starts, default `1.0` |

Example with 200 images:

```bash
python src/capture_burst.py \
    --device /dev/video2 \
    --output-dir data/raw/no_gesture/session_A \
    --fps 5 \
    --num-images 200 \
    --delay 1
```

---

# Dataset

The training data consists of two domains.

## OLD

Selected samples from the original HaGRID dataset.

The three target gestures are taken from their corresponding HaGRID classes.

The OLD `no_gesture` pool contains both:

```text
HaGRID no_gesture samples
+
samples from non-target HaGRID gestures
```

The latter act as hard negatives.

## NEW

Custom indoor recordings containing the colored glove.

A recording session represents one coherent combination of conditions such as:

- location
- background
- lighting
- camera
- camera position
- distance
- viewing angle
- clothing
- left/right hand

Sessions remain intact when splitting the data.

This prevents nearly identical images from one burst appearing in both training and validation.

---

# Dataset structure

```text
Dataset/
│
├── gesture_1/
│   ├── old/
│   ├── session_A/
│   ├── session_B/
│   ├── ...
│   └── session_Z/
│
├── gesture_2/
│   ├── old/
│   ├── session_A/
│   ├── ...
│   └── session_Z/
│
├── gesture_3/
│   ├── old/
│   ├── session_A/
│   ├── ...
│   └── session_Z/
│
└── no_gesture/
    ├── old/
    ├── session_A/
    ├── session_B/
    ├── ...
    └── session_Z/
```

Every top-level directory is automatically interpreted as one class.

Current class mapping:

```text
0: gesture_1
1: gesture_2
2: gesture_3
3: no_gesture
```

---

# Dataset size

The current dataset contains:

## NEW

Per session:

```text
gesture_1       100
gesture_2       100
gesture_3       100
no_gesture      200

Total           500
```

Across 26 sessions:

```text
gesture_1       2,600
gesture_2       2,600
gesture_3       2,600
no_gesture      5,200

Total          13,000
```

## OLD

```text
gesture_1       2,600
gesture_2       2,600
gesture_3       2,600
no_gesture      5,200

Total          13,000
```

Total physical image pool:

```text
OLD            13,000
NEW            13,000
               ------
Total          26,000
```

---

# NEW dataset split

NEW data is split strictly by session:

```text
A-R    Training
S-V    Validation
W-Z    Test
```

This gives:

```text
18 training sessions
 4 validation sessions
 4 test sessions
```

Sample counts:

```text
NEW train       9,000
NEW val         2,000
NEW test        2,000
```

The final test sessions `W-Z` must remain untouched during:

- training
- hyperparameter tuning
- checkpoint selection
- diagnostic analysis

They are used only for the final evaluation.

---

# OLD dataset split

OLD data is split independently per class:

```text
90% OLD training
10% OLD validation
```

Current counts:

```text
OLD train      11,700
OLD val         1,300
```

The split is deterministic.

Default seed:

```text
42
```

---

# Preprocessing

`src/preprocessing.py` converts raw image directories into disk-based model input images.

It:

- recursively scans the input directory
- preserves the relative directory structure
- applies EXIF orientation
- converts images to RGB
- handles transparency
- resizes to `224 × 224`
- stores processed images as JPEG
- gives images timestamp-based filenames
- never modifies the original input files

Basic command:

```bash
python src/preprocessing.py \
    --input-dir data/raw \
    --output-dir data/processed
```

## Preprocessing flags

| Flag | Meaning | Default |
|---|---|---|
| `--input-dir` | source directory | required |
| `--output-dir` | destination directory | required |
| `--image-size` | square image size | `224` |
| `--mode` | resize strategy | `fit-pad` |
| `--quality` | JPEG quality | `90` |

Available resize modes:

```text
fit-pad
center-crop
stretch
```

Explicit example:

```bash
python src/preprocessing.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --image-size 224 \
    --mode fit-pad \
    --quality 90
```

The output directory must not be located inside the input directory.

---

# Tensor preprocessing and normalization

The JPEG images stored on disk are ordinary RGB images.

They are **not** stored as normalized Float32 tensors.

During loading, `dataset.py` performs:

```text
JPEG
  ↓
RGB
  ↓
ToTensor
  ↓
float32
  ↓
pixel range 0...255 → 0...1
  ↓
ImageNet / HaGRID normalization
  ↓
[3, 224, 224]
```

Normalization:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

This normalization is part of the data pipeline, not the exported ResNet itself.

Any future inference implementation must therefore reproduce the same preprocessing before passing data to the ONNX model.

---

# Inspecting the dataset

Before training:

```bash
python src/dataset.py \
    --data-dir /path/to/Dataset
```

Expected current output totals:

```text
OLD train      11700
OLD val         1300
NEW train       9000
NEW val         2000
NEW test        2000
```

The script also prints the per-class counts.

This should always be run after changing the dataset.

---

# HaGRID checkpoint test

Before training, the pretrained checkpoint can be tested independently:

```bash
python src/model.py \
    --checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --num-classes 4
```

The current HaGRID checkpoint contains:

```text
MODEL_STATE
```

and loads:

```text
120 HaGRID backbone tensors
```

Expected architecture summary:

```text
Frozen:
  conv1
  bn1
  layer1
  layer2
  layer3

Trainable:
  layer4
  classifier
```

Outputs:

```text
logits
embedding [512]
```

---

# End-to-end smoke test

Before starting a long training run, run:

```bash
python src/smoke_test.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth
```

The smoke test verifies one real training batch through the complete pipeline:

```text
Dataset
  ↓
Balanced OLD/NEW sampler
  ↓
DataLoader
  ↓
CUDA
  ↓
ResNet-18
  ↓
CrossEntropyLoss
  ↓
Backward
```

It checks:

- `[128, 3, 224, 224]` input shape
- `float32` input
- exactly `64 OLD + 64 NEW`
- valid labels
- logits shape `[128, num_classes]`
- embedding shape `[128, 512]`
- frozen parameters receive no gradients
- `layer4 + classifier` receive gradients
- finite loss
- finite gradients
- frozen BatchNorm layers stay frozen

The smoke test performs:

```text
NO optimizer step
NO checkpoint write
```

A successful run ends with:

```text
SMOKE TEST PASSED
```

---

# Training augmentation

Augmentation is applied **on-the-fly** and only to training samples.

The current V2 baseline uses approximately:

```text
Rotation          ±10°
Translation       ±7%
Scale             0.85 – 1.0
Brightness        ±20%
Contrast          ±20%
Saturation        ±15%
Hue               ±3%
Gaussian Blur     p = 0.15
Gaussian Noise    p = 0.15
```

Validation and test images are never augmented.

Horizontal flipping is disabled by default.

If left and right hand orientation have identical semantic meaning, enable:

```bash
--horizontal-flip
```

Example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --output-dir checkpoints/training/v3 \
    --horizontal-flip
```

The V3 augmentation configuration is intentionally not documented yet because it has not been finalized.

---

# Training sampling

Every batch contains:

```text
Batch size = 128

64 OLD
64 NEW
```

There are:

```text
141 batches / epoch
```

Therefore:

```text
18,048 samples / epoch

9,024 OLD
9,024 NEW
```

Every epoch is exactly:

```text
50% OLD
50% NEW
```

Class balancing is enforced over the complete epoch.

The classes therefore receive approximately equal sampling probability even though the physical dataset contains twice as many `no_gesture` images.

If a class does not contain enough unique images for its required quota, the sampler cycles and reuses samples.

---

# Fine-tuning configuration

Current V2 training configuration:

```text
Architecture           ResNet-18
Initialization         pretrained HaGRID checkpoint

Frozen                 conv1 – layer3
Trainable              layer4 + classifier

Precision              FP32

Batch size             128
Batches / epoch        141
Samples / epoch        18,048

OLD / NEW              50% / 50%

Optimizer              AdamW

LR layer4              3e-5
LR classifier          1.5e-4

Weight decay           1e-4

Loss                   CrossEntropyLoss

Warm-up                3 epochs
Warm-up start          10% target LR

Scheduler              cosine decay

Maximum epochs         30

Gradient clipping      max norm = 1.0

Early stopping         patience = 5
```

Warm-up starts at:

```text
layer4:
3e-6 → 3e-5

classifier:
1.5e-5 → 1.5e-4
```

After warm-up, cosine decay is used.

---

# Precision

Training is intentionally **FP32-only**.

FP16 Automatic Mixed Precision was tested on the development GPU:

```text
NVIDIA T600 Laptop GPU
```

FP32 produced finite logits and loss:

```text
FP32 logits finite: True
FP32 loss finite:   True
```

FP16 AMP produced:

```text
FP16 logits: NaN
FP16 loss:   NaN
```

Therefore the current `train.py` does not use:

```text
autocast
GradScaler
--disable-amp
```

and training is performed directly in FP32.

If mixed precision is reconsidered in the future, it must first pass the complete smoke test on the target hardware.

---

# Starting a fresh training run

Required:

1. processed dataset
2. pretrained HaGRID ResNet-18 checkpoint
3. new output directory

Example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --output-dir checkpoints/training/v3
```

A fresh experiment must start from the original HaGRID checkpoint.

Do **not** use an older experiment's `last.pt` when starting a new hyperparameter experiment.

---

# train.py flags

## `--data-dir`

Required.

```bash
--data-dir /path/to/Dataset
```

Root directory containing:

```text
gesture_1/
gesture_2/
gesture_3/
no_gesture/
```

## `--hagrid-checkpoint`

Required for a fresh run.

```bash
--hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth
```

Not required for resume.

## `--output-dir`

Required.

```bash
--output-dir checkpoints/training/v3
```

Contains:

```text
best.pt
last.pt
```

## `--resume`

Resume from an existing `last.pt`.

```bash
--resume checkpoints/training/v3/last.pt
```

## `--num-workers`

Number of DataLoader worker processes.

Default:

```text
4
```

Example:

```bash
--num-workers 8
```

For debugging DataLoader issues:

```bash
--num-workers 0
```

## `--seed`

Default:

```text
42
```

Example:

```bash
--seed 42
```

## `--horizontal-flip`

Boolean flag.

It takes no value:

```bash
--horizontal-flip
```

Correct:

```bash
python src/train.py ... --horizontal-flip
```

Incorrect:

```text
--horizontal-flip true
```

When the flag is absent, horizontal flipping is disabled.

---

# Boolean CLI flags

Flags created with `action="store_true"` are enabled simply by writing the flag.

Example:

```bash
--horizontal-flip
```

You do not write:

```text
--horizontal-flip True
```

The same general rule applies to other boolean command-line switches.

---

# Training output

After every epoch, `train.py` prints:

```text
Train loss
Train accuracy

OLD validation loss
OLD validation accuracy

NEW validation loss
NEW validation accuracy

LR layer4
LR classifier

early-stopping counter
```

Example:

```text
Epoch 02/30
  Train loss:       0.6380
  Train accuracy:   75.44%
  OLD val loss:     0.1128
  OLD val accuracy: 97.08%
  NEW val loss:     1.1559
  NEW val accuracy: 53.85%
  LR layer4:        ...
  LR classifier:    ...
  Early stopping:   0/5
  BEST:             yes
```

Model selection is based exclusively on:

```text
NEW validation accuracy
```

OLD validation accuracy is diagnostic only.

---

# Checkpoints

Every training experiment creates:

```text
best.pt
last.pt
```

## best.pt

Contains the model state corresponding to the highest observed:

```text
NEW validation accuracy
```

Use `best.pt` for:

```text
diagnosis
final evaluation
ONNX export
```

## last.pt

Contains the complete latest training state:

- model state
- optimizer state
- scheduler state
- current epoch
- best NEW validation accuracy
- early-stopping counter
- class mapping
- training configuration
- RNG state

This allows an interrupted run to continue.

---

# Resume training

Example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v3 \
    --resume checkpoints/training/v3/last.pt
```

When resuming:

```text
model
optimizer
scheduler
epoch
best validation result
early stopping state
RNG state
```

are restored.

The pretrained HaGRID checkpoint is not needed because the full fine-tuned model is already stored in `last.pt`.

Resume should only be used to continue the **same experiment**.

It should not be used to create V2, V3, etc.

---

# Early stopping

Early stopping monitors only:

```text
NEW validation accuracy
```

Patience:

```text
5 epochs
```

Example:

```text
best epoch
↓
no improvement
1/5
2/5
3/5
4/5
5/5
↓
stop
```

`best.pt` remains the best observed checkpoint even when later epochs become worse.

---

# Diagnosis

`src/diagnose.py` is used during development.

It evaluates the selected `best.pt` on:

```text
NEW train A-R
NEW validation S-V
```

using deterministic evaluation preprocessing and **no augmentation**.

It prints:

- overall NEW train accuracy
- NEW train accuracy per class
- NEW train confusion matrix
- overall NEW validation accuracy
- NEW validation accuracy per class
- NEW validation confusion matrix
- generalization gap
- results for S, T, U and V independently
- per-class results for each validation session

Run:

```bash
python src/diagnose.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v3/best.pt
```

Optional flags:

```bash
--batch-size 128
--num-workers 4
```

Example:

```bash
python src/diagnose.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v3/best.pt \
    --batch-size 128 \
    --num-workers 4
```

The diagnosis script must never evaluate `W-Z`.

---

# Understanding the diagnosis

A useful comparison is:

```text
NEW train accuracy
vs.
NEW validation accuracy
```

For example:

```text
NEW train      95%
NEW val        55%
```

indicates a large session-generalization gap.

The confusion matrices then show which class causes the failure.

A particularly important failure mode for this project is:

```text
true gesture
→ predicted no_gesture
```

This means the model recognizes the input as belonging to the negative class instead of preserving the gesture identity under the new recording condition.

---

# Experiment directories

Every independent experiment should use its own output directory.

Example:

```text
checkpoints/training/
│
├── v1/
│   ├── best.pt
│   └── last.pt
│
├── v2/
│   ├── best.pt
│   └── last.pt
│
└── v3/
    ├── best.pt
    └── last.pt
```

This prevents accidentally overwriting previous results.

A new experiment:

```bash
--output-dir checkpoints/training/v3
```

A resumed V3 experiment:

```bash
--output-dir checkpoints/training/v3 \
--resume checkpoints/training/v3/last.pt
```

---

# Current development status

This section is temporary development documentation.

## V1

Configuration:

```text
LR layer4       1e-4
LR classifier   5e-4
```

Best checkpoint:

```text
epoch 2
NEW val accuracy: 53.85%
```

Diagnostic result:

```text
NEW train: 70.51%
NEW val:   53.85%

gap:       16.66 percentage points
```

## V2

Configuration:

```text
LR layer4       3e-5
LR classifier   1.5e-4
```

Best checkpoint:

```text
epoch 10
NEW val accuracy: 53.25%
```

Diagnostic result:

```text
NEW train: 90.46%
NEW val:   53.25%

gap:       37.21 percentage points
```

Validation by session:

```text
S   56.60%
T   44.20%
U   46.80%
V   65.40%
```

Interpretation:

V2 fits the NEW training sessions much more strongly than V1, but validation performance does not improve.

The dominant remaining problem is therefore generalization across recording conditions rather than inability to fit the NEW training data.

The next experiment is V3.

The planned focus for V3 is stronger and more targeted augmentation for:

```text
object / hand scale
translation
viewing angle
perspective
exposure / brightness
contrast
motion blur
camera-condition variation
```

The exact V3 augmentation settings should be documented only after they have been finalized.

---

# Final evaluation

Final evaluation uses:

```text
best.pt
```

and exclusively the untouched NEW sessions:

```text
W-Z
```

Run:

```bash
python src/evaluate.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v3/best.pt
```

The evaluation prints:

- total test samples
- overall accuracy
- accuracy per class
- confusion matrix

No training or augmentation is performed.

## Important

Do not repeatedly evaluate W-Z while developing V1, V2, V3, etc.

Doing so would turn the test set into another validation set.

The intended sequence is:

```text
A-R
training

S-V
model development / selection

W-Z
one final evaluation
```

---

# ONNX export

After final model selection:

```bash
python src/export_onnx.py \
    --checkpoint checkpoints/training/v3/best.pt \
    --output models/hulk-hand.onnx
```

The exported model uses fixed input shape:

```text
[1, 3, 224, 224]
```

Meaning:

```text
batch     1
channels  3
height    224
width     224
```

Outputs:

```text
logits
[1, num_classes]

embedding
[1, 512]
```

Batch size is deliberately fixed to `1`.

---

# ONNX Opset 13

The model is explicitly exported as:

```text
ONNX Opset 13
```

The exporter does not trust the export argument alone.

After writing the temporary ONNX model it performs several independent checks.

## 1. ONNX load

```python
onnx.load(...)
```

must succeed.

## 2. ONNX checker

```python
onnx.checker.check_model(...)
```

must succeed.

## 3. Actual Opset inspection

The generated model's:

```text
opset_import
```

is inspected.

The standard ONNX domain must be exactly:

```text
13
```

Otherwise export fails.

## 4. Shape verification

Required shapes:

```text
Input
[1, 3, 224, 224]

Logits
[1, num_classes]

Embedding
[1, 512]
```

## 5. ONNX Runtime

The resulting model must load successfully with ONNX Runtime.

## 6. PyTorch / ONNX numerical comparison

The same dummy input is passed through:

```text
PyTorch
ONNX Runtime
```

Both outputs are checked:

```text
logits
embedding
```

Tolerance:

```text
rtol = 1e-4
atol = 1e-5
```

Any mismatch outside these limits causes export to fail.

## 7. Atomic finalization

The model is first written to a temporary:

```text
.tmp.onnx
```

The final output file is created only after all validation checks succeed.

---

# Complete workflow

```text
Camera / Raw Images
        │
        ▼
capture_burst.py
        │
        ▼
Raw Dataset
        │
        ▼
preprocessing.py
        │
        ▼
Processed Dataset
        │
        ▼
dataset.py
        │
        ▼
Dataset verification
        │
        ▼
model.py
        │
        ▼
HaGRID checkpoint verification
        │
        ▼
smoke_test.py
        │
        ▼
FP32 pipeline verification
        │
        ▼
sampler.py
        │
        ▼
64 OLD + 64 NEW
        │
        ▼
train.py
        │
        ├────► best.pt
        │
        └────► last.pt
                 │
                 ▼
            diagnose.py
                 │
                 ▼
         Train / Val analysis
                 │
                 ▼
        Hyperparameter iteration
                 │
                 ▼
            final best.pt
                 │
                 ▼
            evaluate.py
                 │
                 ▼
          NEW Test W-Z
                 │
                 ▼
          export_onnx.py
                 │
                 ▼
          hulk-hand.onnx
          ONNX Opset 13
```

---

# Dependencies

Install dependencies from:

```bash
pip install -r requirements.txt
```

Current Python dependencies:

```text
torch
torchvision
Pillow
numpy
tqdm
onnx
onnxscript
onnxruntime
opencv-python
```

`opencv-python` is required by `capture_burst.py`.

`onnxscript` is required by the current PyTorch ONNX exporter.

For CUDA-enabled PyTorch, use a PyTorch build appropriate for the local NVIDIA environment.

---

# Useful troubleshooting commands

## Check CUDA from Python

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Inspect the dataset

```bash
python src/dataset.py \
    --data-dir /path/to/Dataset
```

## Test only the model checkpoint

```bash
python src/model.py \
    --checkpoint checkpoints/pretrained/hagrid_resnet18.pth \
    --num-classes 4
```

## Test the full training pipeline without training

```bash
python src/smoke_test.py \
    --data-dir /path/to/Dataset \
    --hagrid-checkpoint checkpoints/pretrained/hagrid_resnet18.pth
```

## Show training arguments

```bash
python src/train.py --help
```

## Show diagnostic arguments

```bash
python src/diagnose.py --help
```

---

# HaGRID

This project builds upon the HaGRID project and its pretrained ResNet-18 model.

HaGRID:

https://github.com/hukenovs/hagrid

HaGRID models:

https://github.com/hukenovs/hagrid-models

The pretrained HaGRID ResNet-18 checkpoint is used as initialization for the backbone before fine-tuning.

The checkpoint itself is not intended to be committed to this repository.

Recommended location:

```text
checkpoints/pretrained/hagrid_resnet18.pth
```

---

# License

The source code in this repository is distributed under the **BSD 3-Clause License**.

HaGRID datasets, pretrained assets and other third-party materials remain subject to their respective licenses and terms.

Using this repository does not relicense third-party datasets or pretrained assets.

Users are responsible for complying with the applicable licenses of any external datasets, model weights or source code used with this project.

---

# Project scope

The scope of `hulk-hand` ends with creation and validation of the trained model:

```text
Training
Evaluation
ONNX Export
Opset 13 Verification
```

Deployment, camera integration, runtime inference applications and downstream systems are intentionally outside the scope of this repository.