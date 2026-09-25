# Third-Party Notices

This project uses third-party datasets and pretrained model assets that are subject to their own license terms.

The BSD 3-Clause License contained in this repository applies only to original source code authored for this project unless explicitly stated otherwise.

## HaGRID

This project uses the HaGRID dataset and pretrained HaGRID model assets as upstream training resources.

Upstream project:

https://github.com/hukenovs/hagrid

The initial pretrained ResNet18 checkpoint used for training was obtained from the HaGRID project:

https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth

The HaGRID dataset and the original pretrained checkpoint are not included in this repository.

HaGRID dataset samples were used during fine-tuning.

The upstream checkpoint was modified by replacing and training the classification head for this project's gesture-classification task. Resulting model artifacts include PyTorch (`.pth`), ONNX (`.onnx`), and TensorFlow Lite (`.tflite`) representations of the fine-tuned model.

### Upstream License

HaGRID is distributed under a project-specific license described by the HaGRID project as a variant of the Creative Commons Attribution-ShareAlike 4.0 International License.

The applicable upstream HaGRID license can be found here:

https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf

The HaGRID dataset and original pretrained model assets remain subject to the HaGRID project license and are not relicensed under this repository's BSD 3-Clause License.

### Derived Model Artifacts

The model artifacts produced by this project from HaGRID material, including the fine-tuned PyTorch checkpoint and its ONNX and TensorFlow Lite representations, are distributed by this project under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.

License:

https://creativecommons.org/licenses/by-sa/4.0/

This licensing applies to the derived model artifacts only. It does not replace or modify the license applicable to the original HaGRID dataset, pretrained checkpoints, or other upstream HaGRID material.

Redistributions or adaptations of these model artifacts must comply with the applicable CC BY-SA 4.0 requirements, including attribution, indication of modifications, and ShareAlike requirements, as well as any upstream obligations that continue to apply from the HaGRID material from which they were derived.

Use of HaGRID does not imply endorsement of this project by the HaGRID authors or rights holders.