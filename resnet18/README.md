# HaGRID ResNet18

## Dataset Layout

```text
dataset/
├── one/
│   ├── ...
│   └── ...
├── two/
│   ├── ...
│   └── ...
├── stop/
│   ├── ...
│   └── ...
└── no_gesture/
    ├── ...
    └── ...
```

Only the top-level folder name is used as the label. Any deeper subfolders are scanned recursively and ignored for labeling.

## Train

```bash
python train_resnet18.py --train-root /path/to/dataset
```

Recommended large-dataset run:

```bash
python train_resnet18.py \
  --train-root /path/to/dataset \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --label-smoothing 0.05 \
  --warmup-ratio 0.05 \
  --amp \
  --no-use-class-weights
```

The trainer uses:
- full fine-tuning by default
- on-the-fly preprocessing
- stratified train/validation split
- cosine LR decay with warmup
- mixed precision when enabled

Outputs are written to `runs/resnet18/` by default, including `ResNet18_finetuned.pth`, `class_names.txt`, and `training_config.json`.

## Export To ONNX

```bash
python export_resnet18_onnx.py --checkpoint runs/resnet18/ResNet18_finetuned.pth --output ResNet18.onnx --opset 13 --num-classes 4
```

## Webcam Inference

```bash
python webcam_resnet18.py --model ResNet18.onnx
```

## Training Plan

1. Train with the large-dataset defaults above.
2. Watch validation accuracy and validation loss.
3. If training is unstable, lower `--lr` to `1e-4`.
4. If overfitting shows up, increase augmentation or raise `--weight-decay` slightly.
5. If `no_gesture` dominates, retry with `--use-class-weights`.
6. Keep the best checkpoint from validation, then export that checkpoint to ONNX opset 13.
