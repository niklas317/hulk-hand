# ResNet18 Gesture Model

## Task

This repository fine-tunes a pretrained HaGRID ResNet18 checkpoint for four
classes. The output order is fixed and must remain consistent across training,
checkpoint metadata, and deployment:

| Index | Class | Required gesture |
| ---: | --- | --- |
| 0 | `one` | One raised finger. |
| 1 | `two` | Two raised fingers. |
| 2 | `stop` | Open palm with the fingers extended. |
| 3 | `no_gesture` | No target hand gesture, including an empty or irrelevant frame. |

The gesture definitions describe the intended labels, not a claim that every
possible hand pose is covered by the model.

## Training

The base model is the 34-class HaGRID ResNet18 checkpoint downloaded from the
URL below. The final fully connected layer is replaced with a four-output
classifier. `scripts/train_resnet18.py` supports both full fine-tuning and
classifier-only training with `--freeze-backbone`.

Download the base checkpoint into `resnet18/artifacts/`:

```bash
mkdir -p resnet18/artifacts
curl -L "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth" \
  -o resnet18/artifacts/ResNet18.pth
```

The direct training pipeline:

1. Recursively scans `one/`, `two/`, `stop/`, and `no_gesture/` directories.
2. Splits one dataset into stratified training and validation subsets unless
   `--val-root` is supplied.
3. Applies augmentation only to training images.
4. Letterboxes images to 224x224, converts them to RGB tensors, and applies
   the configured normalization.
5. Uses AdamW, warmup, cosine learning-rate decay, optional label smoothing,
   optional class weighting, and optional CUDA AMP.

The default output is `resnet18/artifacts/runs/resnet18/`. The best validation
checkpoint is saved as `ResNet18_finetuned.pth`, alongside `class_names.txt`
and `training_config.json`.

## Input And Outputs

The model input is an RGB float32 NCHW tensor with shape `[batch, 3, 224, 224]`.
The exported dual-output model returns:

- `logits`: `[batch, 4]`, in the class order above.
- `embedding`: `[batch, 512]`, the feature vector before the classifier.

Preprocessing is not embedded in ONNX or TFLite. Deployment must use the same
RGB conversion, letterboxing, tensor conversion, and normalization as training.

## Limitations

Validation accuracy from an internal split is not an independent test result.
Performance can change with lighting, camera angle, hand scale, background,
and users. Validate on representative camera frames before deployment.

## Training Command

```bash
python3 resnet18/scripts/train_resnet18.py \
  --train-root /path/to/dataset \
  --epochs 40 \
  --batch-size 64 \
  --amp
```

## Export

```bash
python3 resnet18/scripts/export_resnet18.py \
  --checkpoint resnet18/artifacts/runs/resnet18/ResNet18_finetuned.pth \
  --output-dir resnet18/artifacts/export
python3 resnet18/scripts/check_onnx_opset.py \
  resnet18/artifacts/export/ResNet18_4class_opset13_dual_output.onnx \
  --expect 13
```

TFLite conversion also requires the `onnx2tf` and TensorFlow packages from
`requirements.txt`. TFLite may rename or reorder outputs; identify logits by
shape `[batch, 4]` and the embedding by shape `[batch, 512]` after conversion.
