#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

# Disable xFormers before DINOv2 is imported by torch.hub.
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F


DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vits14"

IMAGE_SIZE = 224
EMBEDDING_DIM = 384

EXPECTED_ATTENTION_BLOCKS = 12

# V9:
# Fine-tune the last eight transformer blocks.
NUM_TRAINABLE_BLOCKS = 8


class Opset13SelfAttention(nn.Module):
    """
    ONNX Opset-13-friendly replacement for DINOv2 attention.

    Keeps the original pretrained:

        qkv
        proj
        proj_drop

    modules and replaces scaled_dot_product_attention()
    with classic attention:

        scores = (Q @ K^T) * scale
        attention = softmax(scores)
        output = attention @ V
    """

    def __init__(
        self,
        source_attention: nn.Module,
    ) -> None:

        super().__init__()

        required_attributes = (
            "dim",
            "num_heads",
            "scale",
            "qkv",
            "attn_drop",
            "proj",
            "proj_drop",
        )

        missing = [
            name
            for name in required_attributes
            if not hasattr(
                source_attention,
                name,
            )
        ]

        if missing:
            raise RuntimeError(
                "Unsupported DINOv2 attention module. "
                "Missing attributes: "
                + ", ".join(missing)
            )

        self.dim = int(
            source_attention.dim
        )

        self.num_heads = int(
            source_attention.num_heads
        )

        self.scale = float(
            source_attention.scale
        )

        self.attn_drop = float(
            source_attention.attn_drop
        )

        # Reuse exact pretrained modules.
        self.qkv = source_attention.qkv
        self.proj = source_attention.proj
        self.proj_drop = source_attention.proj_drop

        if (
            self.dim
            % self.num_heads
            != 0
        ):
            raise RuntimeError(
                "Attention dimension is not "
                "divisible by number of heads."
            )

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:

        batch_size, num_tokens, channels = (
            x.shape
        )

        if channels != self.dim:
            raise RuntimeError(
                "Unexpected attention input dimension.\n"
                f"Expected: {self.dim}\n"
                f"Actual:   {channels}"
            )

        head_dim = (
            channels
            // self.num_heads
        )

        # ------------------------------------------------------
        # QKV
        # ------------------------------------------------------

        qkv = self.qkv(
            x
        )

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            head_dim,
        )

        query, key, value = torch.unbind(
            qkv,
            dim=2,
        )

        # [B, N, H, D] -> [B, H, N, D]

        query = query.transpose(
            1,
            2,
        )

        key = key.transpose(
            1,
            2,
        )

        value = value.transpose(
            1,
            2,
        )

        # ------------------------------------------------------
        # Classic scaled dot-product attention
        # ------------------------------------------------------

        attention_scores = (
            torch.matmul(
                query,
                key.transpose(
                    -2,
                    -1,
                ),
            )
            * self.scale
        )

        if is_causal:

            causal_mask = torch.ones(
                (
                    num_tokens,
                    num_tokens,
                ),
                dtype=torch.bool,
                device=x.device,
            )

            causal_mask = torch.tril(
                causal_mask
            )

            attention_scores = (
                attention_scores.masked_fill(
                    ~causal_mask,
                    torch.finfo(
                        attention_scores.dtype
                    ).min,
                )
            )

        attention = F.softmax(
            attention_scores,
            dim=-1,
        )

        if (
            self.training
            and self.attn_drop > 0.0
        ):

            attention = F.dropout(
                attention,
                p=self.attn_drop,
                training=True,
            )

        output = torch.matmul(
            attention,
            value,
        )

        output = (
            output
            .transpose(
                1,
                2,
            )
            .contiguous()
            .view(
                batch_size,
                num_tokens,
                channels,
            )
        )

        output = self.proj(
            output
        )

        output = self.proj_drop(
            output
        )

        return output


def _is_dinov2_attention(
    module: nn.Module,
) -> bool:
    """
    Identify DINOv2 Attention / MemEffAttention modules.
    """

    class_name = (
        module
        .__class__
        .__name__
    )

    if class_name not in (
        "Attention",
        "MemEffAttention",
    ):
        return False

    required_attributes = (
        "qkv",
        "proj",
        "num_heads",
        "scale",
    )

    return all(
        hasattr(
            module,
            name,
        )
        for name in required_attributes
    )


def replace_attention_for_opset13(
    module: nn.Module,
) -> int:
    """
    Recursively replace DINOv2 attention modules.
    """

    replaced = 0

    for child_name, child_module in list(
        module.named_children()
    ):

        if _is_dinov2_attention(
            child_module
        ):

            replacement = Opset13SelfAttention(
                child_module
            )

            setattr(
                module,
                child_name,
                replacement,
            )

            replaced += 1

        else:

            replaced += replace_attention_for_opset13(
                child_module
            )

    return replaced


class HulkHandDinoV2(nn.Module):
    """
    hulk-hand V9 model.

    Architecture:

        RGB [B,3,224,224]
               |
               v
        DINOv2 ViT-S/14
               |
        blocks 0-3
        FROZEN
               |
        blocks 4-11
        TRAINABLE
               |
        final LayerNorm
        TRAINABLE
               |
               v
        384-D CLS embedding
               |
               v
        Linear(384, num_classes)
        TRAINABLE
               |
               v
             logits
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        num_trainable_blocks: int = NUM_TRAINABLE_BLOCKS,
    ) -> None:

        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be > 0."
            )

        self.num_classes = (
            num_classes
        )

        self.embedding_dim = (
            EMBEDDING_DIM
        )

        # ------------------------------------------------------
        # Load official pretrained DINOv2 ViT-S/14
        # ------------------------------------------------------

        self.backbone = torch.hub.load(
            DINOV2_REPO,
            DINOV2_MODEL,
            pretrained=pretrained,
        )

        # ------------------------------------------------------
        # Replace attention for Opset 13
        # ------------------------------------------------------

        replaced = replace_attention_for_opset13(
            self.backbone
        )

        if (
            replaced
            != EXPECTED_ATTENTION_BLOCKS
        ):

            raise RuntimeError(
                "Unexpected number of DINOv2 "
                "attention modules replaced.\n"
                f"Expected: {EXPECTED_ATTENTION_BLOCKS}\n"
                f"Actual:   {replaced}"
            )

        self.opset13_attention_blocks = (
            replaced
        )

        # ------------------------------------------------------
        # Validate transformer structure
        # ------------------------------------------------------

        if not hasattr(
            self.backbone,
            "blocks",
        ):
            raise RuntimeError(
                "DINOv2 backbone has no 'blocks' attribute."
            )

        if len(
            self.backbone.blocks
        ) != EXPECTED_ATTENTION_BLOCKS:

            raise RuntimeError(
                "Unexpected number of transformer blocks.\n"
                f"Expected: {EXPECTED_ATTENTION_BLOCKS}\n"
                f"Actual:   {len(self.backbone.blocks)}"
            )

        if not hasattr(
            self.backbone,
            "norm",
        ):
            raise RuntimeError(
                "DINOv2 backbone has no final 'norm'."
            )

        if num_trainable_blocks <= 0:
            raise RuntimeError(
                "num_trainable_blocks must be > 0."
            )

        if num_trainable_blocks > len(self.backbone.blocks):
            raise RuntimeError(
                "num_trainable_blocks exceeds the number of DINOv2 blocks."
            )

        self.num_trainable_blocks = int(
            num_trainable_blocks
        )

        # ------------------------------------------------------
        # Freeze complete backbone first
        # ------------------------------------------------------

        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = False

        # ------------------------------------------------------
        # Unfreeze last transformer blocks
        # ------------------------------------------------------

        first_trainable_block = (
            len(self.backbone.blocks)
            - self.num_trainable_blocks
        )

        self.trainable_block_indices = tuple(
            range(
                first_trainable_block,
                len(self.backbone.blocks),
            )
        )

        for block_index in (
            self.trainable_block_indices
        ):

            block = (
                self.backbone.blocks[
                    block_index
                ]
            )

            for parameter in (
                block.parameters()
            ):
                parameter.requires_grad = True

        # ------------------------------------------------------
        # Unfreeze final LayerNorm
        # ------------------------------------------------------

        for parameter in (
            self.backbone.norm.parameters()
        ):
            parameter.requires_grad = True

        # ------------------------------------------------------
        # New classifier
        # ------------------------------------------------------

        self.classifier = nn.Linear(
            EMBEDDING_DIM,
            num_classes,
        )

        self._initialize_classifier()

        # ------------------------------------------------------
        # Set initial module modes
        # ------------------------------------------------------

        self.eval()

    def _initialize_classifier(
        self,
    ) -> None:

        nn.init.trunc_normal_(
            self.classifier.weight,
            std=0.02,
        )

        if (
            self.classifier.bias
            is not None
        ):

            nn.init.zeros_(
                self.classifier.bias
            )

    def train(
        self,
        mode: bool = True,
    ):
        """
        Selective training mode.

        Frozen backbone parts remain in eval mode.

        During training only:

            blocks 10-11
            final backbone norm
            classifier

        are put into training mode.
        """

        super().train(
            mode
        )

        # Entire backbone starts in eval mode.
        self.backbone.eval()

        if mode:

            for block_index in (
                self.trainable_block_indices
            ):

                self.backbone.blocks[
                    block_index
                ].train(
                    True
                )

            self.backbone.norm.train(
                True
            )

            self.classifier.train(
                True
            )

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        features = (
            self.backbone
            .forward_features(
                x
            )
        )

        embedding = features[
            "x_norm_clstoken"
        ]

        logits = self.classifier(
            embedding
        )

        return {
            "logits": logits,
            "embedding": embedding,
        }


def build_model(
    num_classes: int,
    pretrained: bool = True,
    num_trainable_blocks: int = NUM_TRAINABLE_BLOCKS,
) -> HulkHandDinoV2:

    return HulkHandDinoV2(
        num_classes=num_classes,
        pretrained=pretrained,
        num_trainable_blocks=num_trainable_blocks,
    )


def count_parameters(
    model: nn.Module,
) -> tuple[
    int,
    int,
    int,
]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    return (
        total,
        frozen,
        trainable,
    )


def validate_trainable_parameters(
    model: HulkHandDinoV2,
) -> None:
    """
    Ensure only the intended V9 parts are trainable.
    """

    allowed_prefixes = tuple(
        f"backbone.blocks.{block_index}."
        for block_index in model.trainable_block_indices
    ) + (
        "backbone.norm.",
        "classifier.",
    )

    unexpected_trainable = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if not name.startswith(
            allowed_prefixes
        ):
            unexpected_trainable.append(
                name
            )

    if unexpected_trainable:

        raise RuntimeError(
            "Unexpected trainable parameters:\n"
            + "\n".join(
                unexpected_trainable
            )
        )

    frozen_block_count = model.trainable_block_indices[0]

    for block_index in range(
        0,
        frozen_block_count,
    ):

        block = (
            model.backbone.blocks[
                block_index
            ]
        )

        trainable = [
            name
            for name, parameter
            in block.named_parameters()
            if parameter.requires_grad
        ]

        if trainable:

            raise RuntimeError(
                f"Frozen block {block_index} "
                "contains trainable parameters:\n"
                + "\n".join(
                    trainable
                )
            )

    for block_index in (
        model.trainable_block_indices
    ):

        block = (
            model.backbone.blocks[
                block_index
            ]
        )

        if not any(
            parameter.requires_grad
            for parameter in block.parameters()
        ):

            raise RuntimeError(
                f"Block {block_index} "
                "contains no trainable parameters."
            )

    if not any(
        parameter.requires_grad
        for parameter in (
            model.backbone.norm.parameters()
        )
    ):

        raise RuntimeError(
            "Final backbone norm is not trainable."
        )

    if not any(
        parameter.requires_grad
        for parameter in (
            model.classifier.parameters()
        )
    ):

        raise RuntimeError(
            "Classifier is not trainable."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and smoke-test the hulk-hand "
            "DINOv2 ViT-S/14 V9 model."
        )
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help=(
            "Number of output classes. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
        help=(
            "Device for smoke test. "
            "Default: auto"
        ),
    )

    args = parser.parse_args()

    if args.num_classes <= 0:

        raise ValueError(
            "--num-classes must be > 0."
        )

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------

    print()
    print(
        "hulk-hand DINOv2 V9 model"
    )
    print(
        "========================="
    )
    print()

    print(
        f"PyTorch:          "
        f"{torch.__version__}"
    )

    print(
        f"Device:           "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:              "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Backbone:         "
        f"{DINOV2_MODEL}"
    )

    print(
        f"Embedding dim:    "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Classes:          "
        f"{args.num_classes}"
    )

    print(
        "Attention:        "
        "Opset13 classic"
    )

    print(
        "Trainable blocks: "
        f"{', '.join(str(index) for index in model.trainable_block_indices)}"
    )

    print(
        "Final norm:       "
        "trainable"
    )

    print()

    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    print(
        "Loading pretrained DINOv2 backbone..."
    )

    print()

    model = build_model(
        num_classes=args.num_classes,
        pretrained=True,
        num_trainable_blocks=NUM_TRAINABLE_BLOCKS,
    )

    model.to(
        device
    )

    validate_trainable_parameters(
        model
    )

    # ----------------------------------------------------------
    # Parameter counts
    # ----------------------------------------------------------

    (
        total_parameters,
        frozen_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    print(
        f"Attention blocks replaced: "
        f"{model.opset13_attention_blocks}"
    )

    print(
        f"Trainable block indices:    "
        f"{model.trainable_block_indices}"
    )

    print(
        f"Total parameters:           "
        f"{total_parameters:,}"
    )

    print(
        f"Frozen parameters:          "
        f"{frozen_parameters:,}"
    )

    print(
        f"Trainable parameters:       "
        f"{trainable_parameters:,}"
    )

    # ----------------------------------------------------------
    # Verify train/eval behavior
    # ----------------------------------------------------------

    model.train()

    for block_index in range(
        model.trainable_block_indices[0]
    ):

        if (
            model.backbone.blocks[
                block_index
            ].training
        ):

            raise RuntimeError(
                f"Frozen block {block_index} "
                "entered training mode."
            )

    for block_index in (
        model.trainable_block_indices
    ):

        if not (
            model.backbone.blocks[
                block_index
            ].training
        ):

            raise RuntimeError(
                f"Trainable block {block_index} "
                "did not enter training mode."
            )

    if not model.backbone.norm.training:

        raise RuntimeError(
            "Final backbone norm did not "
            "enter training mode."
        )

    if not model.classifier.training:

        raise RuntimeError(
            "Classifier did not enter training mode."
        )

    # ----------------------------------------------------------
    # Forward + backward smoke test
    # ----------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )

    outputs = model(
        dummy_input
    )

    logits = outputs[
        "logits"
    ]

    embedding = outputs[
        "embedding"
    ]

    expected_logits_shape = (
        1,
        args.num_classes,
    )

    expected_embedding_shape = (
        1,
        EMBEDDING_DIM,
    )

    if (
        tuple(logits.shape)
        != expected_logits_shape
    ):

        raise RuntimeError(
            "Unexpected logits shape.\n"
            f"Expected: {expected_logits_shape}\n"
            f"Actual:   {tuple(logits.shape)}"
        )

    if (
        tuple(embedding.shape)
        != expected_embedding_shape
    ):

        raise RuntimeError(
            "Unexpected embedding shape.\n"
            f"Expected: {expected_embedding_shape}\n"
            f"Actual:   {tuple(embedding.shape)}"
        )

    if not torch.isfinite(
        logits
    ).all():

        raise FloatingPointError(
            "Non-finite logits detected."
        )

    if not torch.isfinite(
        embedding
    ).all():

        raise FloatingPointError(
            "Non-finite embedding detected."
        )

    # Artificial loss only to verify gradients.
    loss = logits.sum()

    loss.backward()

    # ----------------------------------------------------------
    # Verify gradient routing
    # ----------------------------------------------------------

    trainable_without_gradient = []

    frozen_with_gradient = []

    for name, parameter in (
        model.named_parameters()
    ):

        if parameter.requires_grad:

            if parameter.grad is None:

                trainable_without_gradient.append(
                    name
                )

            elif not torch.isfinite(
                parameter.grad
            ).all():

                raise FloatingPointError(
                    "Non-finite gradient detected in "
                    f"'{name}'."
                )

        else:

            if parameter.grad is not None:

                frozen_with_gradient.append(
                    name
                )

    if trainable_without_gradient:

        raise RuntimeError(
            "Trainable parameters without gradients:\n"
            + "\n".join(
                trainable_without_gradient
            )
        )

    if frozen_with_gradient:

        raise RuntimeError(
            "Frozen parameters received gradients:\n"
            + "\n".join(
                frozen_with_gradient
            )
        )

    # ----------------------------------------------------------
    # Result
    # ----------------------------------------------------------

    print()

    print(
        f"Input shape:      "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Logits shape:     "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Embedding shape:  "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Logits finite:    "
        f"{bool(torch.isfinite(logits).all())}"
    )

    print(
        f"Embedding finite: "
        f"{bool(torch.isfinite(embedding).all())}"
    )

    print(
        "Backward:         PASS"
    )

    print(
        "Frozen gradients: PASS"
    )

    print()

    print(
        "========================"
    )

    print(
        "MODEL SMOKE TEST PASSED"
    )

    print(
        "========================"
    )

    print()


if __name__ == "__main__":
    main()
