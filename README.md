# Hulk Hand

Hulk Hand is a repository for PyTorch neural-network experiments focused on hand-gesture classification.

The project fine-tunes pretrained vision models for gesture recognition, with a particular focus on gestures performed using a green glove — hence the name **Hulk Hand**.

Detailed information about the ResNet18 model, training pipeline, preprocessing, architecture, and deployment is available in [`resnet18/RESNET18.md`](resnet18/RESNET18.md).

## Demo

An optional webcam demo is included to demonstrate real-time inference using an exported ONNX model.

The demo is not part of the training pipeline and is provided only as a simple example of how the resulting model can be integrated into an application.

Download a compatible ONNX model from the [project releases](https://github.com/niklas317/hulk-hand/releases) and provide its path when starting the viewer:

```bash
python3 demo/webcam-viewer.py \
  --camera /dev/video0 \
  --model /path/to/ResNet18_4class_opset13_dual_output.onnx
```

## AI-Assisted Development

Parts of the source code and project documentation were developed with assistance from **GPT-5.6 Luna by OpenAI**.

AI-generated or AI-assisted code was reviewed, adapted, and integrated specifically for this project.

## License

Original source code authored for this repository is distributed under the **BSD 3-Clause License**.

Third-party datasets, pretrained model weights, and model artifacts derived from third-party material may be subject to separate license terms.

In particular, the ResNet18 model artifacts are derived from the **HaGRID** project and remain subject to the applicable HaGRID license terms.

For attribution, upstream sources, modifications, and third-party licensing information, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
