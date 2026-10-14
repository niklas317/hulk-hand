# hulk-hand

`hulk-hand` contains two independent four-class gesture-classification
pipelines:

- [ResNet18-HaGRID](resnet18-hagrid/RESNET18.md) — the completed legacy pipeline.
- [ResNet18-Custom](resnet18-custom/RESNET18_CUSTOM.md) — from-scratch training on private NEW sessions.
- [ResNet152-Custom Layer3](resnet152-custom-layer3/RESNET152_CUSTOM.md) — ResNet152 with a 1024-dimensional layer3 output.
- [ResNet152-Custom Layer4](resnet152-custom-layer4/RESNET152_CUSTOM.md) — ResNet152 with its native 2048-dimensional layer4 output.
- [DINOv2](dinov2/DINOv2.md) — the completed legacy pipeline.

Both models use `[batch, 3, 224, 224]` inputs and produce four-class logits, but their preprocessing, embeddings, checkpoints, and requirements remain separate.

| Model | Embedding | Status |
|---|---:|---|
| ResNet18-HaGRID | `[batch, 512]` | Completed legacy pipeline |
| ResNet18-Custom | `[batch, 512]` | Private NEW-only training pipeline |
| ResNet152-Custom Layer3 | `[batch, 1024]` | Private NEW-only training pipeline |
| ResNet152-Custom Layer4 | `[batch, 2048]` | Private NEW-only training pipeline |
| DINOv2 | `[batch, 384]` | Completed |

Deployment artifacts are distributed through releases rather than committed
to the source repository.

## Webcam demo

Download a compatible ONNX model from the [project releases](https://github.com/niklas317/hulk-hand/releases), then provide its path and select the matching preprocessing pipeline:

```bash
python3 demo/webcam-viewer.py \
  --camera /dev/video0 \
  --model /path/to/model.onnx \
  --preprocessing <resnet18|resnet18-custom|resnet152-custom-layer3|resnet152-custom-layer4|dinov2>
```

The `--preprocessing` value must match the model: ResNet18 and DINOv2 use different normalization parameters. The default labels are `one`, `two`, `stop`, and `no_gesture`; use `--class-names` or `--class-file` if the artifact uses different labels.

## License

Original source code authored for this repository is distributed under the **MIT License**.

The model artifacts are licensed separately:

- **ResNet18** artifacts are distributed under **CC BY-SA 4.0**, subject to applicable **HaGRID** upstream license terms.
- **DINOv2** artifacts are distributed under the **Apache License 2.0** and incorporate Meta's Apache-licensed pretrained DINOv2 weights.

For attribution, upstream sources, modifications, and third-party licensing information, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## AI-Assisted Development

Parts of the source code and documentation were developed with assistance from GPT-5.6 Luna by OpenAI, and were reviewed and adapted for this project.
