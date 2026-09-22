# Hulk Hand

Hulk Hand is a repository for PyTorch neural-network experiments focused on
hand-classifier models. The project fine-tunes pretrained vision models for
hand gestures performed with a green glove, giving the project its HULK name.

Detailed ResNet18 classes, training, preprocessing, and deployment information
is documented in [`resnet18/RESNET18.md`](resnet18/RESNET18.md).

## Webcam Viewer

The webcam viewer runs live gesture inference with an exported ONNX model.
Download the model from the [project releases](https://github.com/niklas317/hulk-hand/releases)
and provide its path when starting the viewer:

```bash
python3 demo/webcam-viewer.py \
  --device /dev/video0 \
  --model /path/to/ResNet18_4class_opset13_dual_output.onnx
```
