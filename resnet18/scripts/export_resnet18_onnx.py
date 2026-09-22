#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18WithEmbedding(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits = self.backbone.fc(embedding)
        return logits, embedding


def load_checkpoint(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "MODEL_STATE" in checkpoint:
            return checkpoint["MODEL_STATE"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def load_matching_weights(model: ResNet18WithEmbedding, state_dict: Dict[str, torch.Tensor]) -> None:
    model_state = model.backbone.state_dict()
    filtered_state = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("backbone."):
            normalized_key = normalized_key[len("backbone.") :]

        if normalized_key in model_state and model_state[normalized_key].shape == tensor.shape:
            filtered_state[normalized_key] = tensor

    missing_keys = sorted(set(model_state) - set(filtered_state))
    if missing_keys:
        print(f"Skipping {len(missing_keys)} unmatched parameters, including any resized classifier head.")

    model.backbone.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)} matching tensors into the backbone.")


def build_model(num_classes: int = 4) -> ResNet18WithEmbedding:
    return ResNet18WithEmbedding(num_classes=num_classes)


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 13,
) -> None:
    model.eval()
    dummy_input = torch.randn(*input_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )


def verify_exported_opset(output_path: str | Path, expected_opset: int) -> None:
    try:
        import onnx
    except ImportError:
        print(f"Exported {output_path} with opset {expected_opset}; install onnx to verify the file metadata.")
        return

    model = onnx.load(str(output_path))
    exported_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if expected_opset not in exported_opsets:
        raise RuntimeError(f"Expected opset {expected_opset}, found {exported_opsets}")

    print(f"Verified ONNX opset {expected_opset} in {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HaGRID ResNet18 checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="ResNet18.pth", help="Path to the .pth checkpoint")
    parser.add_argument("--output", default="ResNet18.onnx", help="Path to write the ONNX file")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classifier outputs")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    model = build_model(num_classes=args.num_classes)
    state_dict = load_checkpoint(args.checkpoint)
    load_matching_weights(model, state_dict)
    export_to_onnx(model, args.output, opset_version=args.opset)
    verify_exported_opset(args.output, args.opset)


if __name__ == "__main__":
    main()
