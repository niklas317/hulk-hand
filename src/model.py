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


class Opset13SelfAttention(nn.Module):
    """
    ONNX Opset-13-friendly replacement for DINOv2 attention.

    It keeps the original pretrained modules:

        qkv
        proj
        proj_drop

    and replaces only:

        scaled_dot_product_attention(...)

    with the mathematically equivalent classic formulation:

        scores = (Q @ K^T) * scale
        attention = softmax(scores)
        output = attention @ V

    For hulk-hand the DINOv2 backbone is kept in eval mode,
    so attention dropout is inactive during normal use.
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

        # Reuse the exact pretrained modules.
        #
        # No weights are reinitialized here.
        self.qkv = (
            source_attention.qkv
        )

        self.proj = (
            source_attention.proj
        )

        self.proj_drop = (
            source_attention.proj_drop
        )

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
        # QKV projection
        #
        # [B, N, C]
        #   ->
        # [B, N, 3, H, D]
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

        query, key, value = (
            torch.unbind(
                qkv,
                dim=2,
            )
        )

        # ------------------------------------------------------
        # [B, N, H, D]
        #   ->
        # [B, H, N, D]
        # ------------------------------------------------------

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
        #
        # [B,H,N,D] @ [B,H,D,N]
        #   ->
        # [B,H,N,N]
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

        # DINOv2 does not use causal attention for this
        # image-classification backbone, but preserve the
        # argument for compatibility with the original API.
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

        # Match the original attention-dropout behavior.
        if (
            self.training
            and self.attn_drop > 0.0
        ):

            attention = F.dropout(
                attention,
                p=self.attn_drop,
                training=True,
            )

        # ------------------------------------------------------
        # Attention @ V
        #
        # [B,H,N,N] @ [B,H,N,D]
        #   ->
        # [B,H,N,D]
        # ------------------------------------------------------

        output = torch.matmul(
            attention,
            value,
        )

        # ------------------------------------------------------
        # [B,H,N,D]
        #   ->
        # [B,N,H,D]
        #   ->
        # [B,N,C]
        # ------------------------------------------------------

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
    Identify the Attention / MemEffAttention modules
    currently used inside the DINOv2 backbone.
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
        for name
        in required_attributes
    )


def replace_attention_for_opset13(
    module: nn.Module,
) -> int:
    """
    Recursively replace DINOv2 Attention modules.

    Returns the number of replaced modules.
    """

    replaced = 0

    for child_name, child_module in list(
        module.named_children()
    ):

        if _is_dinov2_attention(
            child_module
        ):

            replacement = (
                Opset13SelfAttention(
                    child_module
                )
            )

            setattr(
                module,
                child_name,
                replacement,
            )

            replaced += 1

        else:

            replaced += (
                replace_attention_for_opset13(
                    child_module
                )
            )

    return replaced


class HulkHandDinoV2(nn.Module):
    """
    hulk-hand V5 model.

    Architecture:

        RGB [B,3,224,224]
               |
               v
        DINOv2 ViT-S/14
        frozen backbone
               |
               v
        384-D CLS embedding
               |
               v
        Linear(384, num_classes)
               |
               v
             logits

    The DINOv2 attention implementation is replaced by
    Opset13SelfAttention so that the model does not use
    aten::scaled_dot_product_attention.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
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

        self.backbone = (
            torch.hub.load(
                DINOV2_REPO,
                DINOV2_MODEL,
                pretrained=pretrained,
            )
        )

        # ------------------------------------------------------
        # Replace all DINOv2 attention modules
        # ------------------------------------------------------

        replaced = (
            replace_attention_for_opset13(
                self.backbone
            )
        )

        if (
            replaced
            != EXPECTED_ATTENTION_BLOCKS
        ):
            raise RuntimeError(
                "Unexpected number of DINOv2 "
                "attention modules replaced.\n"
                f"Expected: "
                f"{EXPECTED_ATTENTION_BLOCKS}\n"
                f"Actual:   {replaced}"
            )

        self.opset13_attention_blocks = (
            replaced
        )

        # ------------------------------------------------------
        # Freeze entire pretrained backbone
        # ------------------------------------------------------

        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = False

        self.backbone.eval()

        # ------------------------------------------------------
        # New trainable classifier
        # ------------------------------------------------------

        self.classifier = nn.Linear(
            EMBEDDING_DIM,
            num_classes,
        )

        self._initialize_classifier()

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

        # Set the overall model mode.
        super().train(
            mode
        )

        # But keep pretrained frozen DINOv2
        # permanently in evaluation mode.
        self.backbone.eval()

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
) -> HulkHandDinoV2:

    return HulkHandDinoV2(
        num_classes=num_classes,
        pretrained=pretrained,
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
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
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


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and smoke-test the "
            "Opset-13-compatible hulk-hand "
            "DINOv2 ViT-S/14 model."
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
        "hulk-hand DINOv2 Opset-13 model"
    )
    print(
        "================================"
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
        "xFormers:         disabled"
    )

    print(
        "Attention:        "
        "Opset13 classic"
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
    )

    model.to(
        device
    )

    model.eval()

    # ----------------------------------------------------------
    # Verify attention replacement
    # ----------------------------------------------------------

    remaining_original_attention = [
        name
        for name, module
        in model.backbone.named_modules()
        if _is_dinov2_attention(
            module
        )
    ]

    if remaining_original_attention:

        raise RuntimeError(
            "Original DINOv2 attention modules "
            "remain after replacement:\n"
            + "\n".join(
                remaining_original_attention
            )
        )

    opset_attention_modules = [
        name
        for name, module
        in model.backbone.named_modules()
        if isinstance(
            module,
            Opset13SelfAttention,
        )
    ]

    if (
        len(opset_attention_modules)
        != EXPECTED_ATTENTION_BLOCKS
    ):

        raise RuntimeError(
            "Incorrect number of Opset13 "
            "attention modules.\n"
            f"Expected: "
            f"{EXPECTED_ATTENTION_BLOCKS}\n"
            f"Actual:   "
            f"{len(opset_attention_modules)}"
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
    # Verify frozen backbone
    # ----------------------------------------------------------

    backbone_trainable = [
        name
        for name, parameter
        in model.backbone.named_parameters()
        if parameter.requires_grad
    ]

    if backbone_trainable:

        raise RuntimeError(
            "Backbone contains trainable "
            "parameters:\n"
            + "\n".join(
                backbone_trainable
            )
        )

    classifier_trainable = [
        name
        for name, parameter
        in model.classifier.named_parameters()
        if parameter.requires_grad
    ]

    if not classifier_trainable:

        raise RuntimeError(
            "Classifier contains no "
            "trainable parameters."
        )

    # ----------------------------------------------------------
    # Forward smoke test
    # ----------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():

        outputs = model(
            dummy_input
        )

    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "Model output must be a dict."
        )

    if set(
        outputs.keys()
    ) != {
        "logits",
        "embedding",
    }:

        raise RuntimeError(
            "Unexpected model outputs: "
            f"{sorted(outputs.keys())}"
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
            f"Expected: "
            f"{expected_logits_shape}\n"
            f"Actual:   "
            f"{tuple(logits.shape)}"
        )

    if (
        tuple(embedding.shape)
        != expected_embedding_shape
    ):

        raise RuntimeError(
            "Unexpected embedding shape.\n"
            f"Expected: "
            f"{expected_embedding_shape}\n"
            f"Actual:   "
            f"{tuple(embedding.shape)}"
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