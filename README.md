# hulk-hand

`hulk-hand` is a static hand gesture classification project for indoor camera scenes.

The model receives a complete preprocessed camera frame and predicts one of four classes:

```text
0: gesture_1
1: gesture_2
2: gesture_3
3: no_gesture
```

The relevant gesture is performed with a colored glove.

The project originally started with a pretrained **HaGRID ResNet-18** backbone. After several experiments showed limited generalization to unseen recording sessions, the active model architecture was changed to **DINOv2 ViT-S/14**.

The current development pipeline covers:

- image capture
- image preprocessing
- dataset organization
- OLD / NEW dataset balancing
- session-separated train / validation / test splits
- transfer learning and fine-tuning
- validation and checkpointing
- diagnostic evaluation
- model-development experiments
- ONNX compatibility testing
- strict ONNX Opset 13 verification
- numerical comparison between PyTorch and ONNX Runtime

The current focus is improving generalization to unseen recording sessions.

The model is still under active development and is **not considered deployment-ready yet**.

Deployment, camera integration and downstream runtime applications are outside the scope of this repository.

---

# Current development status

The current active architecture is:

```text
DINOv2 ViT-S/14
```

with the last eight transformer blocks fine-tuned.

The best model observed so far is:

```text
Experiment          V9
Backbone            DINOv2 ViT-S/14
Best epoch          21
NEW val accuracy    85.80%
NEW val loss         0.5539
Train accuracy      94.70%
OLD val accuracy    93.46%
```

Diagnostic evaluation of that checkpoint produced:

```text
NEW train accuracy  94.70%
NEW val accuracy    85.80%

Generalization gap  8.90 percentage points
```

Validation by unseen session:

```text
S    83.80%
T    69.40%
U    73.40%
V    89.20%
```

The current main limitation is therefore still **cross-session generalization**, although V9 improved the gap substantially.

The final NEW test sessions:

```text
W-Z
```

remain untouched.

They must not be used while continuing model development.

---

# Quick command reference

All commands below assume execution from the repository root:

```bash
cd /path/to/hulk-hand
```

Every CLI script can show its available arguments with:

```bash
python src/<script>.py --help
```

## 1. Inspect the dataset

```bash
python src/dataset.py \
    --data-dir /path/to/Dataset
```

Expected totals:

```text
OLD train    11,700
OLD val       1,300

NEW train     9,000
NEW val       2,000
NEW test      2,000
```

## 2. Smoke-test the current DINOv2 model

```bash
python src/model.py \
    --num-classes 4
```

## 3. Build a filtered V10 dataset

```bash
python src/filter_new_images.py \
    --input-dir /path/to/Dataset \
    --output-dir /path/to/Dataset_V10 \
    --overwrite
```

This creates `clean/` and `rejected/` under the output directory.
`clean/` contains all `old/` samples plus the accepted `new/` samples.

The current model smoke test verifies:

```text
DINOv2 ViT-S/14 loads
12 attention blocks are replaced
blocks 10 and 11 are trainable
final norm is trainable
classifier is trainable
forward pass succeeds
backward pass succeeds
frozen parameters receive no gradients
outputs remain finite
```

Expected output shapes:

```text
Input       [1, 3, 224, 224]
Logits      [1, 4]
Embedding   [1, 384]
```

A successful run ends with:

```text
MODEL SMOKE TEST PASSED
```

## 3. Test DINOv2 ONNX Opset 13 compatibility

The standalone development test can be run with:

```bash
python src/onnx-dinov2-test.py
```

This verifies that the modified DINOv2 architecture can be represented as:

```text
ONNX Opset 13
fixed batch size 1
```

and checks the model with ONNX Runtime.

## 4. Start the current training experiment

Current V9 example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v9 \
    --horizontal-flip
```

A fresh DINOv2 run does **not** require a HaGRID checkpoint.

The pretrained DINOv2 backbone is loaded through `torch.hub`.

## 5. Resume an interrupted V9 run

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v9 \
    --resume checkpoints/training/v9/last.pt \
    --horizontal-flip
```

Resume must only be used to continue the same experiment.

Do not resume V6 when starting V9 or another independent experiment.

## 6. Diagnose the current best V6 checkpoint

```bash
python src/diagnose.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v6/best.pt
```

The current diagnosis evaluates:

```text
NEW train A-R
NEW validation S-V
session S
session T
session U
session V
```

using deterministic preprocessing and no augmentation.

It does not touch W-Z.

## 7. Final test evaluation

Final evaluation is intentionally postponed.

The test sessions:

```text
W-Z
```

must only be evaluated after model development and model selection are complete.

The current project is still being optimized, so W-Z should remain untouched.

## 8. Final ONNX export

Architectural DINOv2 Opset-13 compatibility has already been tested successfully.

The final trained checkpoint has not yet been selected for export because model development is still ongoing.

The final target remains:

```text
Input
[1, 3, 224, 224]

Logits
[1, 4]

Embedding
[1, 384]

Batch size
1

ONNX Opset
13
```

---

# Architecture

The active model uses:

```text
DINOv2 ViT-S/14
```

The current fine-tuning configuration is:

```text
Input
[batch, 3, 224, 224]
        │
        ▼
Patch embedding
        │
        ▼
DINOv2 transformer block 0
        │
        ▼
...
        │
        ▼
DINOv2 transformer block 9
        │
        └──────────── FROZEN
        │
        ▼
DINOv2 transformer block 10
        │
        └──────────── TRAINABLE
        │
        ▼
DINOv2 transformer block 11
        │
        └──────────── TRAINABLE
        │
        ▼
Final LayerNorm
        │
        └──────────── TRAINABLE
        │
        ▼
384-D normalized CLS embedding
        │
        ├────────────► embedding output
        │
        ▼
Linear classifier
384 → 4
        │
        └──────────── TRAINABLE
        │
        ▼
logits
```

The current model contains:

```text
Total parameters       22,058,116
Frozen parameters      18,505,344
Trainable parameters    3,552,772
```

The trainable parts are:

```text
backbone.blocks.10
backbone.blocks.11
backbone.norm
classifier
```

Everything before block 10 remains frozen.

---

# Why DINOv2 replaced ResNet-18

The first project versions used a pretrained HaGRID ResNet-18.

Across several experiments, the ResNet model could fit the training sessions but remained around approximately:

```text
53-57% NEW validation accuracy
```

on unseen recording sessions.

Changes to:

```text
learning rate
augmentation
OLD / NEW sampling
```

did not produce the required improvement.

The project therefore moved from:

```text
HaGRID ResNet-18
```

to:

```text
DINOv2 ViT-S/14
```

The first DINOv2 linear-probe experiment still performed poorly.

A major improvement appeared only after allowing the final two DINOv2 transformer blocks to adapt to the target domain.

---

# Single-model inference design

The project deliberately remains a single image-classification model.

The model input is the complete preprocessed camera frame:

```text
camera frame
    │
    ▼
224x224 letterboxed RGB image
    │
    ▼
DINOv2 classifier
    │
    ▼
one of four classes
```

There is no external:

```text
hand detector
landmark detector
glove crop
pose estimator
multi-stage classifier
```

in the model pipeline.

---

# ONNX Opset 13 compatible attention

A strict project requirement is:

```text
ONNX Opset exactly 13
fixed batch size exactly 1
```

The initial DINOv2 export test failed because the PyTorch attention implementation produced:

```text
aten::scaled_dot_product_attention
```

The tested PyTorch ONNX exporter could not export this operator to Opset 13.

The attention calculation was therefore replaced inside `model.py`.

The original DINOv2 attention computation is represented using the classical formulation:

```text
Q
K
V
│
├───────────────┐
│               │
▼               ▼
Q           transpose(K)
│               │
└──── MatMul ───┘
        │
        ▼
      scale
        │
        ▼
     Softmax
        │
        ▼
   MatMul with V
        │
        ▼
      output
```

The pretrained:

```text
qkv projection
output projection
projection dropout
```

modules and their weights are retained.

Only the attention implementation is replaced.

The replacement uses operations that can be represented by the tested Opset-13 exporter, including:

```text
MatMul
Transpose
Reshape
Softmax
```

All 12 DINOv2 transformer attention blocks are replaced when the model is constructed.

The model smoke test verifies:

```text
Attention blocks replaced: 12
```

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
logits
[batch_size, num_classes]

embedding
[batch_size, 384]
```

For the current four-class configuration:

```text
logits
[batch_size, 4]
```

The embedding is the normalized DINOv2 CLS token after the final backbone normalization.

---

# Repository structure

Current project structure:

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
│   ├── export_onnx.py
│   └── onnx-dinov2-test.py
│
├── checkpoints/
│   ├── pretrained/
│   │   └── hagrid_resnet18.pth
│   │
│   └── training/
│       ├── v1/
│       ├── v2/
│       ├── v3/
│       ├── v4/
│       ├── v5/
│       ├── v6/
│       └── v7/
│
└── models/
    └── ...
```

The HaGRID checkpoint is retained for the historical ResNet experiments but is no longer used by the active DINOv2 model.

The dataset itself may be stored outside the repository.

All important dataset and checkpoint paths are supplied through command-line arguments.

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

The preview is horizontally flipped for display only.

Frames written to disk retain their original camera orientation.

Default delay before recording:

```text
1 second
```

Example with an explicit delay:

```bash
python src/capture_burst.py \
    --device /dev/video2 \
    --output-dir data/raw/no_gesture/session_A \
    --fps 5 \
    --num-images 200 \
    --delay 1
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
| `--delay` | delay before burst starts |

---

# Dataset

The dataset consists of two domains:

```text
OLD
NEW
```

## OLD

OLD consists of selected images derived from HaGRID.

The three target classes contain images corresponding to the target gestures.

The OLD `no_gesture` pool contains:

```text
HaGRID no_gesture
+
non-target HaGRID gesture samples
```

The non-target gestures act as hard negatives.

OLD totals:

```text
gesture_1       2,600
gesture_2       2,600
gesture_3       2,600
no_gesture      5,200

Total          13,000
```

## NEW

NEW consists of custom indoor recordings containing the colored glove.

A recording session represents one coherent set of conditions such as:

- location
- background
- lighting
- camera
- camera position
- distance
- viewing angle
- perspective
- clothing
- hand orientation

Sessions remain intact when splitting the dataset.

This prevents images from the same recording burst from appearing in both training and validation.

Per NEW session:

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

Combined physical image pool:

```text
OLD            13,000
NEW            13,000
               ------
Total          26,000
```

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

Every top-level directory is interpreted as one class.

Current class mapping:

```text
0: gesture_1
1: gesture_2
2: gesture_3
3: no_gesture
```

---

# NEW dataset split

NEW data is split strictly by recording session:

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
- architecture experiments
- checkpoint selection
- diagnosis
- validation analysis

They are reserved only for the final evaluation.

---

# OLD dataset split

OLD data is split independently and deterministically per class:

```text
90% training
10% validation
```

Current counts:

```text
OLD train      11,700
OLD val         1,300
```

Default random seed:

```text
42
```

---

# Preprocessing

The current dataset on disk already consists of preprocessed:

```text
224x224 RGB JPEG
```

images.

The original full-resolution source images used to construct the current dataset are no longer required by the training pipeline.

The current stored images use letterboxed resizing.

`src/preprocessing.py` can be used when creating additional processed data.

It:

- recursively scans the input directory
- preserves relative directory structure
- applies EXIF orientation
- converts images to RGB
- handles transparency
- resizes images
- writes JPEG output
- generates timestamp-based filenames
- does not modify the input files

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

Current dataset images were prepared for:

```text
224 x 224
fit-pad / letterbox
```

---

# Tensor preprocessing and normalization

The stored JPEG files are ordinary RGB images.

They are not stored as normalized Float32 tensors.

During loading:

```text
JPEG
  │
  ▼
RGB
  │
  ▼
ToTensor
  │
  ▼
float32
  │
  ▼
pixel range 0...255 → 0...1
  │
  ▼
normalization
  │
  ▼
[3, 224, 224]
```

Normalization:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Normalization is performed by the data pipeline.

It is not embedded into the exported classifier graph.

Any future inference implementation must reproduce the same input preprocessing.

---

# Inspecting the dataset

Before training:

```bash
python src/dataset.py \
    --data-dir /path/to/Dataset
```

Expected totals:

```text
OLD train      11,700
OLD val         1,300
NEW train       9,000
NEW val         2,000
NEW test        2,000
```

The script also prints class counts.

Run this after any dataset modification.

---

# Current training augmentation

Augmentation is applied on-the-fly and only to training samples.

Validation and test images are never augmented.

The current stronger geometric and color augmentation introduced during V3 includes:

```python
transforms.RandomAffine(
    degrees=15,
    translate=(0.12, 0.12),
    scale=(0.70, 1.10),
)

transforms.RandomPerspective(
    distortion_scale=0.15,
    p=0.25,
)

transforms.ColorJitter(
    brightness=0.30,
    contrast=0.30,
    saturation=0.20,
    hue=0.04,
)
```

The pipeline also contains blur and Gaussian-noise augmentation.

The last confirmed values for those unchanged stages are:

```text
Gaussian Blur     p = 0.15
Gaussian Noise    p = 0.15
```

Horizontal flipping is disabled by default.

For experiments where left/right orientation has the same semantic meaning:

```bash
--horizontal-flip
```

V3 and later DINOv2 experiments discussed below use horizontal flipping.

The stronger augmentation was introduced because the difficult validation sessions contain large variation in:

```text
hand scale
distance
translation
perspective
viewing angle
exposure
contrast
motion blur
background
camera placement
```

---

# Training sampling

The active sampler uses:

```text
25% OLD
75% NEW
```

with:

```text
Batch size = 128

32 OLD
96 NEW
```

There are:

```text
141 batches / epoch
```

which produces:

```text
18,048 samples / epoch

4,512 OLD
13,536 NEW
```

The classes are balanced independently inside the OLD and NEW source pools.

This prevents the physical `no_gesture` class size from automatically dominating the sampled training distribution.

If a class does not contain enough unique images for its epoch quota, samples may be reused.

The 25/75 configuration was introduced in V4 and retained for the current DINOv2 experiments.

---

# Precision

Training is intentionally:

```text
FP32 only
```

FP16 Automatic Mixed Precision was tested on:

```text
NVIDIA T600 Laptop GPU
```

The FP32 test produced finite outputs:

```text
FP32 logits finite    True
FP32 loss finite      True
```

The FP16 AMP test produced:

```text
FP16 logits           non-finite
FP16 loss             NaN
```

Therefore the active training pipeline does not use:

```text
autocast
GradScaler
```

All training is performed directly in FP32.

Any future attempt to use mixed precision must first pass numerical smoke testing on the target hardware.

---

# Current fine-tuning configuration

The current V7 experiment uses the same architecture as V6:

```text
Architecture           DINOv2 ViT-S/14

Frozen                 transformer blocks 0-9

Trainable              transformer block 10
                       transformer block 11
                       final LayerNorm
                       classifier

Embedding              384

Precision              FP32

Batch size             128
Batches / epoch        141
Samples / epoch        18,048

OLD / NEW              25% / 75%

Optimizer              AdamW

Backbone LR            5e-6
Classifier LR          1e-4

Weight decay           1e-3

Loss                    CrossEntropyLoss

Warm-up                3 epochs
Warm-up start           10% target LR

Scheduler               cosine decay

Gradient clipping       max norm = 1.0

Maximum epochs          30

Early stopping          patience = 6
```

V7 deliberately uses lower learning rates and stronger weight decay than V6.

The intention is to reduce overfitting and improve unseen-session generalization.

---

# V6 training configuration

V6 currently remains the best-performing experiment.

```text
Architecture           DINOv2 ViT-S/14

Frozen                 transformer blocks 0-9

Trainable              transformer blocks 10-11
                       final LayerNorm
                       classifier

Precision              FP32

OLD / NEW              25% / 75%

Optimizer              AdamW

Backbone LR            1e-5
Classifier LR          2e-4

Weight decay           1e-4

Warm-up                3 epochs

Scheduler               cosine decay

Gradient clipping       1.0

Maximum epochs          30

Early stopping          patience = 5
```

Best V6 checkpoint:

```text
Epoch                   10
NEW val accuracy        78.95%
NEW val loss             0.5795
```

---

# Starting a fresh training run

A current DINOv2 training run requires:

1. the processed dataset
2. a new output directory
3. access to the DINOv2 backbone through `torch.hub` or its local cache

Example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v7 \
    --horizontal-flip
```

A fresh experiment must not reuse a previous experiment's `last.pt`.

For example:

```text
checkpoints/training/
│
├── v5/
│   ├── best.pt
│   └── last.pt
│
├── v6/
│   ├── best.pt
│   └── last.pt
│
└── v7/
    ├── best.pt
    └── last.pt
```

---

# train.py flags

## `--data-dir`

Required.

```bash
--data-dir /path/to/Dataset
```

## `--output-dir`

Required.

```bash
--output-dir checkpoints/training/v7
```

The directory contains:

```text
best.pt
last.pt
```

## `--resume`

Continue an existing experiment:

```bash
--resume checkpoints/training/v7/last.pt
```

Resume must only be used with a checkpoint created by the same training configuration.

## `--num-workers`

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

## `--horizontal-flip`

Boolean flag:

```bash
--horizontal-flip
```

It takes no value.

Correct:

```bash
python src/train.py ... --horizontal-flip
```

Incorrect:

```text
--horizontal-flip true
```

---

# Training output

After every epoch, the active training script prints:

```text
Train loss
Train accuracy

OLD validation loss
OLD validation accuracy

NEW validation loss
NEW validation accuracy

backbone learning rate
classifier learning rate

early-stopping counter
```

Model selection is based exclusively on:

```text
NEW validation accuracy
```

OLD validation accuracy remains diagnostic.

---

# Checkpoints

Each independent training experiment creates:

```text
best.pt
last.pt
```

## `best.pt`

Contains the model state corresponding to the highest observed:

```text
NEW validation accuracy
```

The selected `best.pt` is used for:

```text
diagnosis
final evaluation
final ONNX export
```

after model development is finished.

## `last.pt`

Contains the latest complete training state, including:

- model state
- optimizer state
- scheduler state
- current epoch
- best NEW validation accuracy
- early-stopping counter
- class list
- class mapping
- number of classes
- model identifier
- training configuration
- RNG state

This allows an interrupted training run to continue.

---

# Resume training

Example:

```bash
python src/train.py \
    --data-dir /path/to/Dataset \
    --output-dir checkpoints/training/v7 \
    --resume checkpoints/training/v7/last.pt \
    --horizontal-flip
```

When resuming, the training script restores:

```text
model
optimizer
scheduler
epoch
best validation result
early stopping state
RNG state
```

The full DINOv2 model state is stored in the checkpoint.

Resume should never be used to silently turn one experiment into another.

---

# Diagnosis

`src/diagnose.py` is used during development.

The current DINOv2 diagnosis evaluates:

```text
NEW train A-R
NEW validation S-V
```

using deterministic evaluation preprocessing and no training augmentation.

It prints:

- overall NEW train accuracy
- NEW train accuracy per class
- NEW train confusion matrix
- overall NEW validation accuracy
- NEW validation accuracy per class
- NEW validation confusion matrix
- validation sessions S, T, U and V separately
- per-class results for each validation session
- confusion matrices for each validation session
- generalization gap
- saved vs recomputed validation accuracy

Current V6 command:

```bash
python src/diagnose.py \
    --data-dir /path/to/Dataset \
    --checkpoint checkpoints/training/v6/best.pt
```

The diagnosis script must never evaluate:

```text
W-Z
```

during model development.

---

# Understanding the diagnosis

The main comparison is:

```text
NEW train accuracy
vs.
NEW validation accuracy
```

A large difference indicates poor generalization to unseen recording sessions.

The confusion matrix provides a second level of analysis.

Earlier ResNet experiments showed a particularly strong failure mode:

```text
true gesture
→ predicted no_gesture
```

After switching to DINOv2 and fine-tuning its final blocks, the failure pattern became less dominated by `no_gesture`.

The remaining errors are now more strongly dependent on individual recording sessions.

---

# Model development and experiments

The architecture and training strategy were developed iteratively.

Model selection always uses:

```text
NEW validation
sessions S-V
```

The final NEW test sessions:

```text
W-Z
```

remain outside the development loop.

---

## V1 — HaGRID ResNet-18 baseline

Architecture:

```text
HaGRID ResNet-18

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

Learning rates:

```text
layer4       1e-4
classifier   5e-4
```

Best checkpoint:

```text
Epoch             2
NEW val accuracy  53.85%
NEW val loss       1.1559
```

Diagnosis:

```text
NEW train accuracy    70.51%
NEW val accuracy      53.85%
Generalization gap    16.66 pp
```

Per-class validation:

```text
gesture_1       21.25%
gesture_2       37.00%
gesture_3       41.25%
no_gesture      84.88%
```

Validation sessions:

```text
S    61.60%
T    45.60%
U    40.20%
V    68.00%
```

A dominant failure mode was:

```text
gesture
→ no_gesture
```

---

## V2 — Lower ResNet learning rates

V2 retained the ResNet architecture but reduced its learning rates.

```text
LR layer4       3e-5
LR classifier   1.5e-4
```

Best checkpoint:

```text
Epoch             10
NEW val accuracy  53.25%
```

Diagnosis:

```text
NEW train accuracy    90.46%
NEW val accuracy      53.25%
Generalization gap    37.21 pp
```

Per-class validation:

```text
gesture_1       22.00%
gesture_2       37.75%
gesture_3       45.50%
no_gesture      80.50%
```

Validation sessions:

```text
S    56.60%
T    44.20%
U    46.80%
V    65.40%
```

V2 showed that reducing the learning rate allowed the model to fit the NEW training data much more strongly but did not improve unseen-session validation.

The main problem was therefore not simply an excessive learning rate.

---

## V3 — Stronger targeted augmentation

V3 retained:

```text
HaGRID ResNet-18

LR layer4       3e-5
LR classifier   1.5e-4

OLD / NEW       50% / 50%
```

but strengthened the augmentation.

Main changes:

```python
transforms.RandomAffine(
    degrees=15,
    translate=(0.12, 0.12),
    scale=(0.70, 1.10),
)

transforms.RandomPerspective(
    distortion_scale=0.15,
    p=0.25,
)

transforms.ColorJitter(
    brightness=0.30,
    contrast=0.30,
    saturation=0.20,
    hue=0.04,
)
```

Horizontal flipping was enabled.

The augmentation targeted variation observed in S-V:

```text
scale
distance
translation
perspective
camera angle
brightness
contrast
blur
```

Reported V3 result:

```text
Epoch             7
Train accuracy    78.28%
OLD val accuracy  97.00%
NEW val accuracy  56.10%
NEW val loss       1.1169
```

This improved over V1/V2 only modestly.

---

## V4 — 25/75 OLD/NEW sampling

V4 retained the stronger augmentation and ResNet architecture but changed the source sampling ratio.

Previous:

```text
50% OLD
50% NEW
```

V4:

```text
25% OLD
75% NEW
```

Per batch:

```text
32 OLD
96 NEW
```

Per epoch:

```text
141 batches
18,048 samples

4,512 OLD
13,536 NEW
```

Observed results:

```text
Epoch 1    NEW val 45.60%
Epoch 2    NEW val 49.50%
Epoch 3    NEW val 52.55%
Epoch 4    NEW val 55.40%
Epoch 5    NEW val 56.60%
Epoch 6    NEW val 55.25%
```

Best observed result:

```text
NEW val accuracy  56.60%
NEW val loss       1.0967
```

Increasing the NEW source contribution did not produce a major architectural breakthrough.

At this point the ResNet experiments remained clustered around approximately:

```text
53-57% NEW validation accuracy
```

This motivated a backbone change.

---

# Migration to DINOv2

The next experiments replaced ResNet-18 entirely.

New architecture:

```text
224x224 RGB frame
        │
        ▼
DINOv2 ViT-S/14
        │
        ▼
384-D CLS embedding
        │
        ▼
Linear(384, 4)
        │
        ▼
four classes
```

The dataset and session split remained unchanged.

The 25/75 OLD/NEW source sampling introduced in V4 was retained.

---

## DINOv2 ONNX experiment

Before committing to DINOv2 training, ONNX Opset 13 compatibility was tested.

The first export attempt failed with:

```text
UnsupportedOperatorError:

aten::scaled_dot_product_attention

Support for this operator was added in
ONNX opset version 14.
```

Because Opset 13 is a strict requirement, the model was not accepted in this form.

A custom classical attention implementation was then introduced.

It preserves the pretrained DINOv2 projection weights while replacing the SDPA operation with:

```text
MatMul
scale
Softmax
MatMul
```

After the replacement:

```text
12 / 12 attention blocks replaced
```

and the DINOv2 architecture successfully passed the Opset-13 smoke test.

This compatibility test was completed before committing to the DINOv2 training experiments.

---

## V5 — Frozen DINOv2 linear probe

The first DINOv2 experiment froze the complete backbone.

Only:

```text
Linear(384, 4)
```

was trainable.

Parameter counts:

```text
Total parameters       22,058,116
Frozen parameters      22,056,576
Trainable parameters        1,540
```

Training:

```text
Optimizer          SGD
Classifier LR      0.01
Momentum           0.9
Weight decay       0

OLD / NEW          25% / 75%
Precision          FP32
```

Observed results:

```text
Epoch 1
Train accuracy     55.65%
NEW val accuracy   50.60%

Epoch 2
Train accuracy     62.21%
NEW val accuracy   52.85%

Epoch 3
Train accuracy     62.38%
NEW val accuracy   55.30%

Epoch 4
Train accuracy     63.02%
NEW val accuracy   50.65%
```

Best observed result:

```text
NEW val accuracy   55.30%
```

The frozen generic DINOv2 representation alone was therefore insufficient.

The project then moved from linear probing to partial backbone fine-tuning.

---

## V6 — Fine-tune the final two DINOv2 blocks

V6 changed the DINOv2 training strategy substantially.

Architecture:

```text
blocks 0-9        frozen
block 10          trainable
block 11          trainable
final norm        trainable
classifier        trainable
```

Parameter counts:

```text
Total parameters       22,058,116
Frozen parameters      18,505,344
Trainable parameters    3,552,772
```

Training configuration:

```text
Optimizer              AdamW

Backbone LR            1e-5
Classifier LR          2e-4

Weight decay           1e-4

Warm-up                3 epochs
Scheduler               cosine

Gradient clipping      1.0

OLD / NEW              25% / 75%

Precision              FP32

Maximum epochs         30
Early stopping         patience 5
```

Training progression:

```text
Epoch 1    NEW val 32.35%
Epoch 2    NEW val 63.60%
Epoch 3    NEW val 70.15%
Epoch 4    NEW val 73.75%
Epoch 5    NEW val 76.30%
Epoch 6    NEW val 73.95%
Epoch 7    NEW val 77.30%
Epoch 8    NEW val 78.35%
Epoch 9    NEW val 76.25%
Epoch 10   NEW val 78.95%
```

Best checkpoint:

```text
Epoch                 10

Train accuracy        91.84%
OLD val accuracy      90.46%

NEW val accuracy      78.95%
NEW val loss           0.5795
```

Training continued through epoch 15.

The model did not exceed the epoch-10 result and early stopping triggered.

V6 was the first experiment to produce a major improvement in unseen-session performance.

---

# V6 diagnosis

The best V6 checkpoint was evaluated again using deterministic evaluation preprocessing.

Checkpoint:

```text
checkpoints/training/v6/best.pt
```

The saved and recomputed validation results matched exactly:

```text
Saved NEW val          78.95%
Recomputed NEW val     78.95%
Difference             +0.0000 pp
```

## NEW train A-R

```text
Overall accuracy       95.61%
```

Per class:

```text
gesture_1              96.06%
gesture_2              92.94%
gesture_3              98.11%
no_gesture             95.47%
```

Confusion matrix:

```text
             gesture_1  gesture_2  gesture_3  no_gesture
gesture_1         1729         47          5          19
gesture_2           25       1673         81          21
gesture_3            4         22       1766           8
no_gesture           62         72         29        3437
```

## NEW validation S-V

```text
Overall accuracy       78.95%
```

Per class:

```text
gesture_1              77.75%
gesture_2              76.25%
gesture_3              87.00%
no_gesture             76.88%
```

Confusion matrix:

```text
             gesture_1  gesture_2  gesture_3  no_gesture
gesture_1          311         35          4          50
gesture_2           65        305          4          26
gesture_3           30          5        348          17
no_gesture           74         71         40         615
```

Generalization gap:

```text
95.61% - 78.95%
=
16.66 percentage points
```

---

# V6 validation by session

Each validation session contains:

```text
500 samples
```

Overall accuracy:

```text
Session S    83.80%
Session T    69.40%
Session U    73.40%
Session V    89.20%
```

## Session S

```text
Overall        83.80%

gesture_1      69.00%
gesture_2      93.00%
gesture_3      91.00%
no_gesture     83.00%
```

## Session T

```text
Overall        69.40%

gesture_1      82.00%
gesture_2      59.00%
gesture_3      67.00%
no_gesture     69.50%
```

## Session U

```text
Overall        73.40%

gesture_1      74.00%
gesture_2      58.00%
gesture_3      94.00%
no_gesture     70.50%
```

## Session V

```text
Overall        89.20%

gesture_1      86.00%
gesture_2      95.00%
gesture_3      96.00%
no_gesture     84.50%
```

The remaining error is strongly session-dependent.

Sessions:

```text
T
U
```

are substantially harder than:

```text
S
V
```

This is evidence that the main remaining problem is still generalization to recording conditions rather than one universally failing class.

---

## V7 — More conservative DINOv2 fine-tuning

V7 retained exactly the same model architecture as V6.

Only the optimization configuration was changed.

V7:

```text
Backbone LR            5e-6
Classifier LR          1e-4

Weight decay           1e-3

Warm-up                3 epochs
Scheduler               cosine

Gradient clipping      1.0

Early stopping         patience 6
```

Compared with V6:

```text
                      V6            V7

Backbone LR           1e-5          5e-6
Classifier LR         2e-4          1e-4
Weight decay          1e-4          1e-3
Patience              5             6
```

The intention was:

```text
slower feature adaptation
+
stronger regularization
=
better session generalization
```

Observed results through epoch 12:

```text
Epoch  1    NEW val 25.65%
Epoch  2    NEW val 53.70%
Epoch  3    NEW val 66.05%
Epoch  4    NEW val 70.25%
Epoch  5    NEW val 72.30%
Epoch  6    NEW val 72.90%
Epoch  7    NEW val 75.50%
Epoch  8    NEW val 76.10%
Epoch  9    NEW val 76.55%
Epoch 10    NEW val 76.95%
Epoch 11    NEW val 76.10%
Epoch 12    NEW val 74.85%
```

Best observed V7 result through epoch 12:

```text
Epoch             10
Train accuracy    89.53%
NEW val accuracy  76.95%
NEW val loss       0.5956
```

At the same epoch, V6 had achieved:

```text
Train accuracy    91.84%
NEW val accuracy  78.95%
NEW val loss       0.5795
```

The stronger regularization therefore has not improved validation performance so far.

V6 remains the best observed checkpoint.

---

# Experiment summary

| Version | Backbone | Main change | Best observed NEW val |
|---|---|---|---:|
| V1 | HaGRID ResNet-18 | Initial fine-tuning baseline | 53.85% |
| V2 | HaGRID ResNet-18 | Lower learning rates | 53.25% |
| V3 | HaGRID ResNet-18 | Stronger targeted augmentation | 56.10% |
| V4 | HaGRID ResNet-18 | 25/75 OLD/NEW sampling | 56.60% |
| V5 | DINOv2 ViT-S/14 | Frozen linear probe | 55.30% |
| V6 | DINOv2 ViT-S/14 | Fine-tune final 2 blocks | **78.95%** |
| V7 | DINOv2 ViT-S/14 | Lower LR + stronger regularization | 76.95%* |
| V9 | DINOv2 ViT-S/14 | Fine-tune last 8 blocks + layerwise decay | **85.80%** |

`*` Best value observed through epoch 12.

The largest improvement was produced by:

```text
allowing the final DINOv2 transformer blocks
to adapt to the target domain
```

rather than by:

```text
learning-rate tuning alone
stronger ResNet augmentation alone
sampling changes alone
frozen DINOv2 features alone
```

---

# Development objective

The project is not optimizing only for training accuracy.

The required property is reliable behavior under previously unseen recording conditions.

The main development objective is therefore:

```text
reduce session sensitivity
```

while maintaining high performance for all four classes.

The difficult validation conditions indicate sensitivity to combinations of:

```text
distance
hand scale
perspective
camera angle
lighting
contrast
blur
background
```

A model with approximately 79% validation accuracy is not currently considered sufficiently reliable for deployment.

Development therefore continues before W-Z is opened.

---

# Experiment directories

Every independent experiment must use its own output directory.

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
├── v3/
│   ├── best.pt
│   └── last.pt
│
├── v4/
│   ├── best.pt
│   └── last.pt
│
├── v5/
│   ├── best.pt
│   └── last.pt
│
├── v6/
│   ├── best.pt
│   └── last.pt
│
└── v7/
    ├── best.pt
    └── last.pt
```

This prevents older experiment results from being overwritten.

---

# Final evaluation

Final evaluation must only happen after model development is complete.

It will use:

```text
final selected best.pt
```

and exclusively:

```text
NEW test sessions W-Z
```

The intended split usage is:

```text
A-R
training
    │
    ▼
S-V
model development
validation
architecture selection
hyperparameter selection
    │
    ▼
W-Z
one final test evaluation
```

W-Z must not be repeatedly evaluated while trying new model versions.

Doing so would effectively convert the test set into another validation set.

At the current development stage:

```text
W-Z remain untouched
```

---

# ONNX export target

The final model must use:

```text
ONNX Opset exactly 13
```

with fixed input:

```text
[1, 3, 224, 224]
```

Meaning:

```text
batch       1
channels    3
height      224
width       224
```

Required outputs:

```text
logits
[1, 4]

embedding
[1, 384]
```

Batch size is deliberately fixed to:

```text
1
```

---

# ONNX validation requirements

The export process must not trust the exporter argument alone.

The final ONNX model must pass all of the following.

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

The model's:

```text
opset_import
```

must report exactly:

```text
13
```

for the standard ONNX domain.

## 4. Input shape

Must be exactly:

```text
[1, 3, 224, 224]
```

No dynamic batch dimension is allowed.

## 5. Output shapes

Must be:

```text
logits
[1, 4]

embedding
[1, 384]
```

## 6. ONNX Runtime

The exported model must load successfully with ONNX Runtime.

## 7. Numerical comparison

The same dummy input must be passed through:

```text
PyTorch
ONNX Runtime
```

and both:

```text
logits
embedding
```

must be compared.

Target tolerance:

```text
rtol = 1e-4
atol = 1e-5
```

Any mismatch outside these limits makes the export invalid.

## 8. Atomic finalization

The final ONNX file should only replace the destination after all validation checks pass.

---

# ONNX development status

DINOv2 Opset-13 compatibility was tested before training.

Initial result:

```text
FAIL

aten::scaled_dot_product_attention
requires a newer ONNX opset in the tested exporter
```

After replacing DINOv2 SDPA with classical attention:

```text
PASS
```

The architecture successfully passed:

```text
fixed input validation
logits shape validation
384-D embedding validation
ONNX checker
actual Opset-13 verification
ONNX Runtime loading
PyTorch / ONNX numerical comparison
```

Therefore the current DINOv2 architecture is considered technically compatible with the project's Opset-13 requirement.

The final trained checkpoint export remains a later step after model selection.

---

# Complete development workflow

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
224x224 letterboxed JPEG dataset
        │
        ▼
dataset.py
        │
        ├──────── OLD
        │
        └──────── NEW sessions A-Z
        │
        ▼
Dataset verification
        │
        ▼
sampler.py
        │
        ▼
25% OLD / 75% NEW
        │
        ▼
model.py
        │
        ▼
DINOv2 ViT-S/14
Opset-13-compatible attention
        │
        ▼
Model forward/backward smoke test
        │
        ▼
ONNX Opset-13 smoke test
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
       NEW train A-R analysis
       NEW val S-V analysis
       per-session S/T/U/V analysis
                 │
                 ▼
        Architecture /
        hyperparameter iteration
                 │
                 ▼
        final selected best.pt
                 │
                 ▼
             evaluate.py
                 │
                 ▼
        final NEW test W-Z
                 │
                 ▼
           export_onnx.py
                 │
                 ▼
          hulk-hand.onnx
          fixed batch = 1
          ONNX Opset 13
```

---

# Dependencies

Install project dependencies with:

```bash
pip install -r requirements.txt
```

Current Python dependencies include:

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

`opencv-python` is used by `capture_burst.py`.

ONNX-related packages are used by the export and validation tests.

DINOv2 is loaded using `torch.hub`.

For CUDA-enabled PyTorch, install a build appropriate for the local NVIDIA environment.

Current development environment:

```text
Ubuntu 24.04

PyTorch
2.13.0+cu130

CUDA
13.0

GPU
NVIDIA T600 Laptop GPU
```

---

# Useful troubleshooting commands

## Check CUDA

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

## Test the current model

```bash
python src/model.py \
    --num-classes 4
```

## Test ONNX Opset 13 compatibility

```bash
python src/onnx-dinov2-test.py
```

## Show training arguments

```bash
python src/train.py --help
```

## Show diagnosis arguments

```bash
python src/diagnose.py --help
```

---

# Legacy HaGRID model

The project originally used HaGRID and a pretrained ResNet-18 checkpoint.

The historical checkpoint is stored outside Git or under:

```text
checkpoints/pretrained/hagrid_resnet18.pth
```

The ResNet architecture was used for:

```text
V1
V2
V3
V4
```

and was replaced by DINOv2 starting with V5.

HaGRID:

```text
https://github.com/hukenovs/hagrid
```

HaGRID models:

```text
https://github.com/hukenovs/hagrid-models
```

The HaGRID checkpoint is no longer required by the active DINOv2 training pipeline.

---

# DINOv2

The active backbone is based on the DINOv2 project:

```text
https://github.com/facebookresearch/dinov2
```

Current backbone:

```text
dinov2_vits14
```

The project uses DINOv2 as a pretrained visual backbone and adds its own four-class classifier.

The current architecture additionally replaces the runtime implementation of self-attention with an Opset-13-compatible classical attention calculation while retaining the pretrained projection weights.

DINOv2 code, weights and other upstream assets remain subject to their respective upstream terms and licenses.

---

# License

The source code in this repository is distributed under the **BSD 3-Clause License**.

Third-party datasets, pretrained weights, external source code and model assets remain subject to their respective licenses and terms.

This repository does not relicense:

```text
HaGRID data
HaGRID pretrained assets
DINOv2 code
DINOv2 pretrained weights
other external assets
```

Users are responsible for complying with the applicable terms for all third-party material used with the project.

---

# Project scope

The scope of `hulk-hand` covers model creation and validation:

```text
Image Capture
Dataset Creation
Preprocessing
Training
Validation
Diagnosis
Model Development
Final Evaluation
ONNX Export
Opset 13 Verification
```

The project currently ends at the creation of a validated model artifact.

Deployment systems, live camera integration, downstream applications and production runtime integration are intentionally outside the scope of this repository.

The model is still under active development.

The current best validation result is not considered sufficiently reliable for deployment, and development will continue before the final W-Z test set is opened.
