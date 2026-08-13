#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18


class HulkHandResNet18(nn.Module):
    """
    ResNet-18 initialized from a pretrained HaGRID checkpoint.

    Frozen:
        conv1
        bn1
        layer1
        layer2
        layer3

    Trainable:
        layer4
        classifier

    Outputs:
        logits:    [batch_size, num_classes]
        embedding: [batch_size, 512]
    """

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2."
            )

        self.num_classes = num_classes
        self.embedding_dim = 512

        backbone = resnet18(
            weights=None
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool

        # New classifier for the target gesture classes.
        self.classifier = nn.Linear(
            self.embedding_dim,
            num_classes,
        )

        self._freeze_early_backbone()

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_early_backbone(
        self,
    ) -> None:
        """
        Freeze all backbone parameters before layer4.
        """

        frozen_modules = (
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
        )

        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

        for parameter in self.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

        self._set_frozen_modules_eval()

    def _set_frozen_modules_eval(
        self,
    ) -> None:
        """
        Keep BatchNorm statistics of frozen layers fixed.
        """

        self.bn1.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()

    def train(
        self,
        mode: bool = True,
    ) -> HulkHandResNet18:
        """
        Keep frozen BatchNorm layers in eval mode even
        when the complete model enters training mode.
        """

        super().train(mode)

        if mode:
            self._set_frozen_modules_eval()

        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return the 512-dimensional penultimate embedding.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        embedding = torch.flatten(
            x,
            1,
        )

        return embedding

    def forward(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:

        embedding = self.forward_features(
            x
        )

        logits = self.classifier(
            embedding
        )

        return {
            "logits": logits,
            "embedding": embedding,
        }

    # ------------------------------------------------------------------
    # HaGRID checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, Tensor]:
        """
        Extract model weights from supported checkpoint formats.

        Supported formats include:

            MODEL_STATE
            state_dict
            raw state_dict
        """

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                "Checkpoint must be a dictionary."
            )

        # Current HaGRID v2 checkpoint format.
        if (
            "MODEL_STATE" in checkpoint
            and isinstance(
                checkpoint["MODEL_STATE"],
                dict,
            )
        ):
            state_dict = checkpoint[
                "MODEL_STATE"
            ]

        # Common PyTorch checkpoint format.
        elif (
            "state_dict" in checkpoint
            and isinstance(
                checkpoint["state_dict"],
                dict,
            )
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        # Raw state_dict.
        else:
            state_dict = checkpoint

        cleaned_state_dict: dict[str, Tensor] = {}

        for key, value in state_dict.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):
                continue

            clean_key = key

            # Compatibility with wrapped checkpoints.
            prefixes = (
                "module.",
                "model.",
            )

            changed = True

            while changed:

                changed = False

                for prefix in prefixes:

                    if clean_key.startswith(
                        prefix
                    ):
                        clean_key = clean_key[
                            len(prefix):
                        ]

                        changed = True

            cleaned_state_dict[
                clean_key
            ] = value

        if not cleaned_state_dict:
            raise RuntimeError(
                "No model tensors found in checkpoint."
            )

        return cleaned_state_dict

    def load_hagrid_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """
        Load the pretrained HaGRID ResNet-18 backbone.

        HaGRID classification heads are intentionally
        ignored.

        Only:

            conv1
            bn1
            layer1
            layer2
            layer3
            layer4

        are transferred.
        """

        checkpoint_path = (
            Path(checkpoint_path)
            .expanduser()
            .resolve()
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "HaGRID checkpoint not found: "
                f"{checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        state_dict = (
            self._extract_state_dict(
                checkpoint
            )
        )

        backbone_prefixes = (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
            "layer3.",
            "layer4.",
        )

        model_state = self.state_dict()

        expected_keys = [
            key
            for key in model_state
            if key.startswith(
                backbone_prefixes
            )
        ]

        backbone_state: dict[
            str,
            Tensor,
        ] = {}

        missing_keys: list[str] = []
        shape_mismatches: list[str] = []

        for key in expected_keys:

            if key not in state_dict:

                missing_keys.append(
                    key
                )

                continue

            checkpoint_tensor = (
                state_dict[key]
            )

            model_tensor = (
                model_state[key]
            )

            if (
                checkpoint_tensor.shape
                != model_tensor.shape
            ):

                shape_mismatches.append(
                    f"{key}: "
                    f"checkpoint="
                    f"{tuple(checkpoint_tensor.shape)}, "
                    f"model="
                    f"{tuple(model_tensor.shape)}"
                )

                continue

            backbone_state[
                key
            ] = checkpoint_tensor

        if missing_keys:

            raise RuntimeError(
                "HaGRID checkpoint is missing "
                "required backbone parameters:\n"
                + "\n".join(
                    missing_keys
                )
            )

        if shape_mismatches:

            raise RuntimeError(
                "HaGRID checkpoint contains "
                "incompatible backbone shapes:\n"
                + "\n".join(
                    shape_mismatches
                )
            )

        self.load_state_dict(
            backbone_state,
            strict=False,
        )

        self._freeze_early_backbone()

        print(
            f"Loaded {len(backbone_state)} "
            "HaGRID backbone tensors."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def trainable_parameters(
        self,
    ):
        return (
            parameter
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    def parameter_summary(
        self,
    ) -> dict[str, int]:

        total = sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

        frozen = (
            total
            - trainable
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


def build_model(
    num_classes: int,
    checkpoint_path: str | Path,
) -> HulkHandResNet18:
    """
    Build hulk-hand and initialize its backbone
    from the pretrained HaGRID ResNet-18.
    """

    model = HulkHandResNet18(
        num_classes=num_classes
    )

    model.load_hagrid_checkpoint(
        checkpoint_path
    )

    return model


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and inspect the hulk-hand "
            "ResNet-18."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the pretrained HaGRID "
            "ResNet-18 checkpoint."
        ),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help=(
            "Number of target gesture classes."
        ),
    )

    args = parser.parse_args()

    model = build_model(
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
    )

    summary = (
        model.parameter_summary()
    )

    print()
    print("hulk-hand ResNet-18")
    print("-------------------")

    print(
        f"Classes:        "
        f"{model.num_classes}"
    )

    print(
        f"Embedding:      "
        f"{model.embedding_dim}"
    )

    print(
        f"Total params:   "
        f"{summary['total']:,}"
    )

    print(
        f"Frozen params:  "
        f"{summary['frozen']:,}"
    )

    print(
        f"Trainable:      "
        f"{summary['trainable']:,}"
    )

    print()
    print("Frozen:")
    print("  conv1")
    print("  bn1")
    print("  layer1")
    print("  layer2")
    print("  layer3")

    print()
    print("Trainable:")
    print("  layer4")
    print("  classifier")

    print()
    print("Outputs:")
    print("  logits")
    print("  embedding [512]")
    print()


if __name__ == "__main__":
    main()