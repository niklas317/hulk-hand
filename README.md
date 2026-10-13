# hulk-hand

`hulk-hand` contains two independent four-class gesture-classification
pipelines:

- [ResNet18](resnet18/RESNET18.md) — the completed legacy pipeline.
- [DINOv2](dinov2/DINOv2.md) — the current development model.

Both models use `[batch, 3, 224, 224]` inputs and produce four-class logits,
but their preprocessing, embeddings, checkpoints, and requirements remain
separate.

| Model | Embedding | Status |
|---|---:|---|
| ResNet18 | `[batch, 512]` | Completed |
| DINOv2 | `[batch, 384]` | Current development model |

Deployment artifacts are distributed through releases rather than committed
to the source repository.

## Webcam demo

Download a compatible ONNX model from the
[project releases](https://github.com/niklas317/hulk-hand/releases), then
provide its path and select the matching preprocessing pipeline:

```bash
python3 demo/webcam-viewer.py \
  --camera /dev/video0 \
  --model /path/to/model.onnx \
  --preprocessing <resnet18|dinov2>
```

The `--preprocessing` value must match the model: ResNet18 and DINOv2 use
different normalization parameters. The default labels are `one`, `two`,
`stop`, and `no_gesture`; use `--class-names` or `--class-file` if the
artifact uses different labels.

## License

Original source code authored for this repository is distributed under the
**BSD 3-Clause License**.

Third-party datasets, pretrained model weights, and model artifacts derived
from third-party material may be subject to separate license terms.

In particular, the ResNet18 model artifacts are derived from the **HaGRID**
project and remain subject to the applicable HaGRID license terms. For
attribution, upstream sources, modifications, and third-party licensing
information, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## AI-Assisted Development

Parts of the source code and documentation were developed with assistance from
GPT-5.6 Luna by OpenAI, and were reviewed and adapted for this project.
