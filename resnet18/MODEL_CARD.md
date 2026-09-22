# Model Card: ResNet18 Gesture Classifier — 4 Classes

## Overview

This model is a fine-tuned ResNet18 gesture classifier for four gesture classes:

1. `one`
2. `two`
3. `stop`
4. `no_gesture`

The model was initialized from a pretrained checkpoint originally coming from the **HaGRID Models repository**. The original checkpoint was trained for **34 gesture classes**. For this project, the model was adapted to a reduced four-class task by replacing the final classification layer and training only the new final layer while keeping the pretrained backbone frozen.

## Model Lineage

- **Base architecture:** ResNet18
- **Original source:** HaGRID Models repository
- **Original task:** 34-class hand gesture classification
- **Fine-tuned task:** 4-class gesture classification
- **Fine-tuning dataset:** Project-local compressed/preprocessed dataset derived from HaGRID V2
- **Fine-tuning method:** Frozen backbone, newly initialized 4-class classification head
- **Best validation accuracy:** 99.07 %

## Training Setup

The training used a pretrained ResNet18 backbone and replaced the original 34-class fully connected output layer with a new 4-class output layer.

During the fine-tuning run:

- The ResNet18 backbone was frozen.
- Only the final classification layer was trained.
- The dataset was preprocessed into a local, simplified image dataset.
- Images were resized/prepared to `224 × 224`.
- Training used the same compressed local dataset structure generated for this repository.
- Validation was performed using an internal validation split.

The reported validation accuracy is based on this internal validation split and should not be interpreted as performance on a fully independent external test set.

## Classes

The model outputs logits for the following class order:

| Index | Class |
|---:|---|
| 0 | `one` |
| 1 | `two` |
| 2 | `stop` |
| 3 | `no_gesture` |

The class order should be kept consistent with `class_names.txt`.

## Input

Expected model input:

```text
float32 tensor
shape: [batch, 3, 224, 224]
color format: RGB
layout: NCHW
```

The model expects images to be preprocessed in the same way as during training/evaluation.

Typical preprocessing pipeline:

1. Load image or camera frame.
2. Convert to RGB.
3. Resize or pad/crop to `224 × 224`, matching the preprocessing used for the local compressed dataset.
4. Convert to float tensor.
5. Apply the same normalization used during training.
6. Add batch dimension.

Preprocessing is **not embedded** into the exported ONNX or TFLite model unless a separate wrapper is added externally.

## Outputs

The exported dual-output model returns two outputs:

| Output | Shape | Description |
|---|---:|---|
| `logits` | `[batch, 4]` | Raw class logits for the four gesture classes |
| `embedding` | `[batch, 512]` | Penultimate-layer feature embedding before the final fully connected layer |

The predicted class can be obtained by applying `argmax` over the `logits` output.

The `embedding` output can be used for downstream tasks such as similarity search, debugging, clustering, visualization, or additional lightweight classifiers.

## Exported Artifacts

Expected project artifacts:

```text
resnet18_4class_head/
├── ResNet18_finetuned.pth
├── class_names.txt
├── training_config.json
└── export/
    ├── ResNet18_4class_opset13_dual_output.onnx
    └── ResNet18_4class_dual_output.tflite
```

The PyTorch checkpoint file contains the fine-tuned model weights. The ONNX and TFLite exports are intended for deployment/inference.

## ONNX Export

The ONNX model should be exported with:

```text
ONNX opset: 13
Outputs: logits, embedding
```

The ONNX model should be checked after export to confirm that the main `ai.onnx` opset import is actually `13`.

## TFLite Export

The TFLite model is derived from the exported dual-output model. Tensor names may differ after conversion, so output order should be verified after export.

Expected logical outputs:

1. `logits`
2. `embedding`

## Intended Use

This model is intended for recognizing the following hand gesture states:

- `one`
- `two`
- `stop`
- `no_gesture`

It is suitable for local inference in applications where these four gestures need to be detected from preprocessed image frames.

## Out-of-Scope Use

This model is not intended for:

- General 34-class HaGRID gesture recognition.
- Robust gesture recognition for arbitrary camera setups without validation.
- Medical, safety-critical, legal, or biometric identity applications.
- Use without matching the expected preprocessing pipeline.
- Deployment in environments with substantially different lighting, camera angle, hand scale, or background distribution unless further tested.

## Limitations

- The reported 99.07 % validation accuracy comes from the project’s validation split, not necessarily from a fully independent test set.
- Performance may drop on images that differ strongly from the fine-tuning data distribution.
- The model only recognizes four classes and cannot distinguish the original 34 HaGRID classes.
- If the preprocessing differs from training, inference quality may degrade significantly.
- The `no_gesture` class quality depends heavily on how representative the negative examples are.

## Recommended Validation Before Deployment

Before using the model in production or a demo, validate it on a small manually reviewed test set from the actual target environment.

Recommended checks:

- Confusion matrix across the four classes.
- Accuracy on real camera frames.
- False positives for `no_gesture`.
- Robustness to lighting changes.
- Robustness to different users/hands/backgrounds.
- Latency on the target hardware.
- Correct output ordering in ONNX/TFLite runtime.

## Version Notes

- Fine-tuned model type: ResNet18, 4-class classifier
- Base checkpoint: HaGRID 34-class ResNet18 checkpoint
- Fine-tuning strategy: frozen backbone, train final layer only
- Export target: ONNX opset 13 and TFLite
- Additional exported feature: penultimate-layer embedding

## Reproducibility Notes

The final model depends on:

- The original HaGRID Models checkpoint.
- The local compressed/preprocessed HaGRID V2-derived dataset.
- The train/validation split used during fine-tuning.
- The exact preprocessing and normalization settings.
- The fine-tuning script configuration.

Keep the following files together when archiving the model:

```text
ResNet18_finetuned.pth
class_names.txt
training_config.json
ResNet18_4class_opset13_dual_output.onnx
ResNet18_4class_dual_output.tflite
MODEL_CARD.md
```

## License and Attribution

The base model checkpoint originates from the HaGRID Models repository, and the fine-tuning data is derived from HaGRID V2. When distributing this model or derived artifacts, retain attribution to the original HaGRID project and comply with the applicable dataset/model licenses.
