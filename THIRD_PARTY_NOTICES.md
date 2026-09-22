# Third-Party Notices

This project uses third-party datasets and pretrained model assets that are subject to their own license terms.

The BSD 3-Clause License contained in this repository applies only to original source code authored for this project unless explicitly stated otherwise.

## HaGRID

The ResNet18 model provided with this project is derived from the HaGRID project.

Upstream project:

https://github.com/hukenovs/hagrid

The initial pretrained ResNet18 checkpoint was obtained from the HaGRID project:

https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth

HaGRID dataset samples were additionally used during fine-tuning.

The upstream checkpoint was modified by fine-tuning it for this project's gesture-classification task. Resulting model artifacts may include PyTorch (`.pth`), ONNX (`.onnx`) and TensorFlow Lite (`.tflite`) representations of the fine-tuned model.

HaGRID is distributed under the HaGRID project license, which is based on the Creative Commons Attribution-ShareAlike 4.0 International License but constitutes a separate project-specific license.

The applicable HaGRID license can be found here:

https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf

HaGRID data, pretrained model assets, and model artifacts derived from them are not relicensed under this repository's BSD 3-Clause License. They remain subject to the applicable HaGRID license terms.

When redistributing or further modifying these model artifacts, users are responsible for complying with the HaGRID license, including applicable attribution, modification-notice, and licensing requirements.

Use of HaGRID does not imply endorsement of this project by the HaGRID authors or rights holders.
