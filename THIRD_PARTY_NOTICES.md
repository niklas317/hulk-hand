# Third-Party Notices

Original source code authored for hulk-hand is licensed under the **MIT License**, as provided in the repository's root `LICENSE` file. This MIT license applies to original project source code, not automatically to third-party materials or model artifacts.

The project's DINOv2-derived model artifacts are distributed under the **Apache License, Version 2.0**, subject to the upstream notices and obligations described below. The root MIT source-code license does not replace the Apache 2.0 terms applicable to those model artifacts.

Third-party materials retain their applicable licenses. In particular, the HaGRID-derived ResNet18 model artifacts are distributed separately under **CC BY-SA 4.0**, with additional upstream HaGRID license conditions that may apply. The root MIT license does **not** relicense the ResNet18 artifacts or any third-party dataset.

## HaGRID (ResNet18 pipeline)

This project uses the HaGRID dataset and pretrained HaGRID model assets as upstream training resources for its ResNet18 pipeline.

Upstream project:

https://github.com/hukenovs/hagrid

The initial pretrained ResNet18 checkpoint used for training was obtained from the HaGRID project:

https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth

The HaGRID dataset and the original pretrained checkpoint are not included in this repository.

HaGRID dataset samples were used during ResNet18 fine-tuning.

The upstream checkpoint was modified by replacing and training the classification head for this project's gesture-classification task. Resulting model artifacts include PyTorch (`.pth`), ONNX (`.onnx`), and TensorFlow Lite (`.tflite`) representations of the fine-tuned model.

### Upstream License

HaGRID is distributed under a project-specific license described by the HaGRID project as a variant of the Creative Commons Attribution-ShareAlike 4.0 International License.

The applicable upstream HaGRID license can be found here:

https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf

The HaGRID dataset and original pretrained model assets remain subject to the HaGRID project license and are not relicensed under this repository's MIT License.

### Derived Model Artifacts

The model artifacts produced by this project from HaGRID material, including the fine-tuned PyTorch checkpoint and its ONNX and TensorFlow Lite representations, are distributed by this project under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.

License:

https://creativecommons.org/licenses/by-sa/4.0/

This licensing applies to the derived ResNet18 model artifacts only. It does not replace or modify the license applicable to the original HaGRID dataset, pretrained checkpoints, or other upstream HaGRID material.

Redistributions or adaptations of these model artifacts must comply with the applicable CC BY-SA 4.0 requirements, including attribution, indication of modifications, and ShareAlike requirements, as well as any upstream obligations that continue to apply from the HaGRID material from which they were derived.

Use of HaGRID does not imply endorsement of this project by the HaGRID authors or rights holders.

## DINOv2 (Meta AI)

The DINOv2 pipeline fine-tunes a pretrained vision-transformer backbone created by Meta AI:

- Upstream project: https://github.com/facebookresearch/dinov2
- Model: **DINOv2 ViT-S/14** (without register tokens)
- Torch Hub identifier: `dinov2_vits14`
- Upstream pretrained weights:
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
- Upstream model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- Upstream license: **Apache License, Version 2.0**
- Upstream license text: https://github.com/facebookresearch/dinov2/blob/main/LICENSE

The DINOv2 model card identifies the original model's license as Apache 2.0. The pretrained backbone is obtained using PyTorch's `torch.hub` functionality; the original pretrained checkpoint is not committed to this source repository.

### Changes Made by hulk-hand

The pretrained DINOv2 ViT-S/14 backbone has been adapted to classify four hand gesture classes. In this adaptation:

- Transformer blocks 0 and 1 remain frozen, while blocks 2 through 11 and the final normalization layer are fine-tuned.
- A new four-class linear classifier is trained.
- Attention computation is replaced with a classical implementation compatible with ONNX Opset 13 while preserving the underlying attention parameters.
- Optional export-time ONNX graph rewrites support deployment compatibility.

Fine-tuning uses a separately collected custom image dataset. (No HaGRID training images or pretrained HaGRID weights are used in the DINOv2 pipeline.) The private training dataset is not distributed in this repository or in its model releases.

### Derived DINOv2 Model Artifacts

The project's original contributions to the fine-tuned DINOv2 model and its distributed model artifacts are licensed under **Apache License 2.0**. These artifacts incorporate Meta's pretrained model parameters, which remain subject to their upstream Apache 2.0 terms and applicable notices.

This covers the fine-tuned PyTorch checkpoint and its corresponding exported formats (such as ONNX and, if provided, TensorFlow Lite). Converting or quantizing an artifact does not remove applicable upstream license obligations.

Recipients may use, modify, and redistribute these artifacts, including in commercial and proprietary products, under the terms of Apache License 2.0. Redistributors must provide a copy of the applicable Apache 2.0 license, retain applicable copyright and attribution notices, identify modifications as required by the license, and carry forward relevant upstream `NOTICE` contents if any were provided.

Neither Meta Platforms, Inc. nor the upstream DINOv2 authors endorse this project.