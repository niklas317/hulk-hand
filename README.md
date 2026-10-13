# hulk-hand

`hulk-hand` contains two independent four-class gesture-classification
pipelines:

- [ResNet18](resnet18/RESNET18.md) — the completed legacy pipeline.
- [DINOv2](dinov2/DINOv2.md) — the current best development model.

Both models use complete preprocessed camera frames and produce four-class
logits. Their preprocessing, embeddings, checkpoints, and requirements remain
separate.

## Model comparison

| Model | Input | Logits | Embedding | Status |
|---|---|---:|---:|---|
| ResNet18 | `[batch, 3, 224, 224]` | `[batch, 4]` | `[batch, 512]` | Completed |
| DINOv2 | `[batch, 3, 224, 224]` | `[batch, 4]` | `[batch, 384]` | Current development model |

Deployment artifacts are distributed through releases rather than committed
to the source repository. The webcam viewer is shared, but the correct class
names and preprocessing must be selected for the model being used.

## Webcam demo

Download an ONNX model from the project release, then provide its path and
select the matching preprocessing pipeline:

```bash
python demo/webcam-viewer.py \
    --model /path/to/model.onnx \
    --preprocessing <resnet18|dinov2>
```

The `--preprocessing` value must match the model. ResNet18 and DINOv2 use
different normalization parameters. The default displayed class names are
`one`, `two`, `stop`, and `no_gesture`; use `--class-names` or `--class-file`
if the release artifact uses different labels.
