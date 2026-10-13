# Export compatibility variants

These scripts modify only the exported ONNX graph. The trained checkpoint is
never changed and no retraining is required for the initial tests.

Run one variant:

```bash
python src/exporter/export_variants.py \
    --checkpoint checkpoints/training/v11/best.pt \
    --output-dir models/onnx/v11 \
    --variant baseline
```

Available variants:

- `baseline`: current export, no compatibility rewrite
- `gelu`: replace exact Erf-based GELU with tanh GELU
- `pow`: replace `Pow(x, 2)` with `Mul(x, x)`
- `resize`: replace cubic Resize with linear Resize
- `all`: export all four variants

Every result is checked for ONNX validity, Opset 13, fixed input shape, and
the expected logits/embedding outputs. PyTorch and ONNX Runtime maximum
absolute differences are printed for comparison.
