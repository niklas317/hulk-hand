#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import build_model
from sampler import (
    BATCH_SIZE,
    BATCHES_PER_EPOCH,
    SAMPLES_PER_SOURCE_PER_BATCH,
    build_training_dataset_and_sampler,
)


def print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def verify_parameter_freezing(model: nn.Module) -> None:
    """
    Only layer4 and classifier may be trainable.
    """

    errors = []

    for name, parameter in model.named_parameters():
        expected_trainable = (
            name.startswith("layer4.")
            or name.startswith("classifier.")
        )

        if parameter.requires_grad != expected_trainable:
            errors.append(
                f"{name}: requires_grad={parameter.requires_grad}, "
                f"expected={expected_trainable}"
            )

    if errors:
        raise RuntimeError(
            "Incorrect parameter freezing:\n"
            + "\n".join(errors)
        )


def verify_frozen_batchnorm(model: nn.Module) -> None:
    """
    BatchNorm layers inside frozen modules must remain in eval mode.
    """

    frozen_modules = [
        model.bn1,
        model.layer1,
        model.layer2,
        model.layer3,
    ]

    errors = []

    for module in frozen_modules:
        for submodule in module.modules():
            if isinstance(submodule, nn.BatchNorm2d):
                if submodule.training:
                    errors.append(
                        "Frozen BatchNorm layer is in training mode."
                    )

    if errors:
        raise RuntimeError(
            "\n".join(errors)
        )


def verify_gradients(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Verify:
      - frozen parameters received no gradients
      - trainable parameters received gradients
      - trainable gradients are finite
    """

    frozen_with_grad = []
    trainable_without_grad = []
    non_finite_grad = []

    frozen_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:
            trainable_params += parameter.numel()

            if parameter.grad is None:
                trainable_without_grad.append(name)
                continue

            if not torch.isfinite(parameter.grad).all():
                non_finite_grad.append(name)

        else:
            frozen_params += parameter.numel()

            if parameter.grad is not None:
                frozen_with_grad.append(name)

    if frozen_with_grad:
        raise RuntimeError(
            "Frozen parameters received gradients:\n"
            + "\n".join(frozen_with_grad)
        )

    if trainable_without_grad:
        raise RuntimeError(
            "Trainable parameters received no gradients:\n"
            + "\n".join(trainable_without_grad)
        )

    if non_finite_grad:
        raise RuntimeError(
            "Non-finite gradients detected:\n"
            + "\n".join(non_finite_grad)
        )

    return frozen_params, trainable_params


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FP32 end-to-end smoke test for the "
            "hulk-hand training pipeline."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the preprocessed Dataset directory.",
    )

    parser.add_argument(
        "--hagrid-checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained HaGRID ResNet-18 checkpoint.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        args.hagrid_checkpoint
        .expanduser()
        .resolve()
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # ==========================================================
    # CUDA
    # ==========================================================

    print_header("CUDA")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
    print(f"CUDA version:  {torch.version.cuda}")
    print("Precision:     FP32")

    # ==========================================================
    # Dataset
    # ==========================================================

    print_header("Dataset")

    datasets, classes, class_to_idx = build_datasets(
        data_dir,
        seed=args.seed,
    )

    required_dataset_keys = {
        "old_train",
        "old_val",
        "new_train",
        "new_val",
        "new_test",
    }

    missing = (
        required_dataset_keys
        - set(datasets.keys())
    )

    if missing:
        raise RuntimeError(
            "Missing datasets: "
            + ", ".join(sorted(missing))
        )

    old_train = datasets["old_train"]
    old_val = datasets["old_val"]
    new_train = datasets["new_train"]
    new_val = datasets["new_val"]
    new_test = datasets["new_test"]

    num_classes = len(classes)

    print()
    print("Classes:")

    for class_name, class_idx in sorted(
        class_to_idx.items(),
        key=lambda item: item[1],
    ):
        print(
            f"  {class_idx}: {class_name}"
        )

    print()
    print(f"OLD train:  {len(old_train)}")
    print(f"OLD val:    {len(old_val)}")
    print(f"NEW train:  {len(new_train)}")
    print(f"NEW val:    {len(new_val)}")
    print(f"NEW test:   {len(new_test)}")

    if num_classes <= 1:
        raise RuntimeError(
            f"Invalid number of classes: {num_classes}"
        )

    if len(old_train) == 0:
        raise RuntimeError(
            "OLD training dataset is empty."
        )

    if len(new_train) == 0:
        raise RuntimeError(
            "NEW training dataset is empty."
        )

    # ==========================================================
    # Sampler
    # ==========================================================

    print_header("Sampler")

    training_dataset, batch_sampler = (
        build_training_dataset_and_sampler(
            old_train,
            new_train,
            seed=args.seed,
        )
    )

    if len(batch_sampler) != BATCHES_PER_EPOCH:
        raise RuntimeError(
            f"Expected {BATCHES_PER_EPOCH} batches, "
            f"got {len(batch_sampler)}."
        )

    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(0)

    print(f"Batch size:       {BATCH_SIZE}")
    print(
        f"OLD per batch:    "
        f"{SAMPLES_PER_SOURCE_PER_BATCH}"
    )
    print(
        f"NEW per batch:    "
        f"{SAMPLES_PER_SOURCE_PER_BATCH}"
    )
    print(
        f"Batches / epoch:  "
        f"{len(batch_sampler)}"
    )

    loader = DataLoader(
        training_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # ==========================================================
    # Load one real training batch
    # ==========================================================

    print_header("Training batch")

    batch = next(iter(loader))

    required_batch_keys = {
        "image",
        "label",
        "class_name",
        "source",
        "session",
        "path",
    }

    missing = (
        required_batch_keys
        - set(batch.keys())
    )

    if missing:
        raise RuntimeError(
            "Batch is missing keys: "
            + ", ".join(sorted(missing))
        )

    images = batch["image"]
    labels = batch["label"]

    expected_image_shape = (
        BATCH_SIZE,
        3,
        224,
        224,
    )

    if tuple(images.shape) != expected_image_shape:
        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(images.shape)}; "
            f"expected {expected_image_shape}"
        )

    if tuple(labels.shape) != (BATCH_SIZE,):
        raise RuntimeError(
            f"Unexpected label shape: "
            f"{tuple(labels.shape)}"
        )

    if images.dtype != torch.float32:
        raise RuntimeError(
            f"Expected float32 images, "
            f"got {images.dtype}"
        )

    if labels.dtype != torch.int64:
        raise RuntimeError(
            f"Expected int64 labels, "
            f"got {labels.dtype}"
        )

    if not torch.isfinite(images).all():
        raise RuntimeError(
            "Input contains NaN or Inf."
        )

    if labels.min().item() < 0:
        raise RuntimeError(
            "Negative class label detected."
        )

    if labels.max().item() >= num_classes:
        raise RuntimeError(
            "Class label exceeds number of classes."
        )

    # ==========================================================
    # OLD / NEW balance
    # ==========================================================

    sources = [
        str(source).strip().lower()
        for source in batch["source"]
    ]

    source_counts = Counter(sources)

    old_count = source_counts.get(
        "old",
        0,
    )

    new_count = source_counts.get(
        "new",
        0,
    )

    if old_count != SAMPLES_PER_SOURCE_PER_BATCH:
        raise RuntimeError(
            f"Expected "
            f"{SAMPLES_PER_SOURCE_PER_BATCH} OLD samples, "
            f"got {old_count}."
        )

    if new_count != SAMPLES_PER_SOURCE_PER_BATCH:
        raise RuntimeError(
            f"Expected "
            f"{SAMPLES_PER_SOURCE_PER_BATCH} NEW samples, "
            f"got {new_count}."
        )

    if old_count + new_count != BATCH_SIZE:
        raise RuntimeError(
            "Unexpected source values in batch: "
            f"{dict(source_counts)}"
        )

    class_counts = Counter(
        labels.tolist()
    )

    print(
        f"Image shape:      "
        f"{tuple(images.shape)}"
    )

    print(
        f"Image dtype:      "
        f"{images.dtype}"
    )

    print(
        f"Input min:        "
        f"{images.min().item():.6f}"
    )

    print(
        f"Input max:        "
        f"{images.max().item():.6f}"
    )

    print(
        f"OLD / NEW:        "
        f"{old_count} / {new_count}"
    )

    print(
        "Class counts:     "
        + ", ".join(
            f"{class_to_idx[class_name]}="
            f"{class_counts[class_to_idx[class_name]]}"
            for class_name in classes
        )
    )

    # ==========================================================
    # Model
    # ==========================================================

    print_header("Model")

    model = build_model(
        num_classes,
        checkpoint_path,
    )

    model = model.to(
        device
    )

    model.train()

    verify_parameter_freezing(
        model
    )

    verify_frozen_batchnorm(
        model
    )

    print(
        "Parameter freezing:  OK"
    )

    print(
        "Frozen BatchNorm:     OK"
    )

    # ==========================================================
    # Optimizer
    #
    # Created to reproduce the real training setup.
    # optimizer.step() is intentionally NOT called.
    # ==========================================================

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.classifier.parameters(),
                "lr": 5e-4,
            },
        ],
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    images = images.to(
        device,
        non_blocking=True,
    )

    labels = labels.to(
        device,
        non_blocking=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # ==========================================================
    # FP32 Forward
    # ==========================================================

    print_header("FP32 Forward + Backward")

    outputs = model(
        images
    )

    if not isinstance(outputs, dict):
        raise RuntimeError(
            "Model must return a dict."
        )

    if "logits" not in outputs:
        raise RuntimeError(
            "Model output has no 'logits'."
        )

    if "embedding" not in outputs:
        raise RuntimeError(
            "Model output has no 'embedding'."
        )

    logits = outputs["logits"]
    embedding = outputs["embedding"]

    expected_logits_shape = (
        BATCH_SIZE,
        num_classes,
    )

    expected_embedding_shape = (
        BATCH_SIZE,
        512,
    )

    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"Unexpected logits shape: "
            f"{tuple(logits.shape)}; "
            f"expected {expected_logits_shape}"
        )

    if tuple(embedding.shape) != expected_embedding_shape:
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{tuple(embedding.shape)}; "
            f"expected {expected_embedding_shape}"
        )

    if logits.dtype != torch.float32:
        raise RuntimeError(
            f"Expected FP32 logits, got {logits.dtype}"
        )

    if embedding.dtype != torch.float32:
        raise RuntimeError(
            f"Expected FP32 embedding, got {embedding.dtype}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Non-finite values detected in logits."
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            "Non-finite values detected in embedding."
        )

    loss = criterion(
        logits,
        labels,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss.item()}"
        )

    # ==========================================================
    # FP32 Backward
    # ==========================================================

    loss.backward()

    grad_norm = clip_grad_norm_(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        max_norm=1.0,
    )

    if not torch.isfinite(
        torch.as_tensor(grad_norm)
    ):
        raise RuntimeError(
            "Non-finite gradient norm."
        )

    frozen_params, trainable_params = (
        verify_gradients(
            model
        )
    )

    print(
        f"Logits:           "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Embedding:        "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Logits dtype:     "
        f"{logits.dtype}"
    )

    print(
        f"Logits min:       "
        f"{logits.min().item():.6f}"
    )

    print(
        f"Logits max:       "
        f"{logits.max().item():.6f}"
    )

    print(
        f"Loss:             "
        f"{loss.item():.6f}"
    )

    print(
        f"Gradient norm:    "
        f"{float(grad_norm):.6f}"
    )

    print(
        f"Frozen params:    "
        f"{frozen_params:,}"
    )

    print(
        f"Trainable params: "
        f"{trainable_params:,}"
    )

    print(
        "Gradient check:   OK"
    )

    print(
        "FP32:             OK"
    )

    # ==========================================================
    # Cleanup
    #
    # Intentionally:
    #   - no optimizer.step()
    #   - no checkpoint
    # ==========================================================

    optimizer.zero_grad(
        set_to_none=True
    )

    del outputs
    del logits
    del embedding
    del loss
    del images
    del labels

    torch.cuda.empty_cache()

    print()
    print("==========================")
    print("SMOKE TEST PASSED")
    print("==========================")
    print()
    print(
        "Dataset -> sampler -> CUDA -> model -> "
        "FP32 loss -> backward works."
    )
    print(
        "No optimizer step was performed."
    )
    print(
        "No checkpoint was written."
    )


if __name__ == "__main__":
    main()