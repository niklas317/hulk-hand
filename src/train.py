#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from model import (
    DINOV2_MODEL,
    EMBEDDING_DIM,
    HulkHandDinoV2,
    build_model,
)
from sampler import (
    BATCH_SIZE,
    BATCHES_PER_EPOCH,
    SAMPLES_PER_EPOCH,
    build_training_dataset_and_sampler,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MODEL_ID = "dinov2_vits14_linear_probe_opset13"

MAX_EPOCHS = 30

CLASSIFIER_LR = 1e-2

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0

EARLY_STOPPING_PATIENCE = 5

VALIDATION_BATCH_SIZE = 128

PRECISION = "fp32"

SOURCE_SPLIT = "25% OLD / 75% NEW"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def seed_worker(
    worker_id: int,
) -> None:
    """
    Seed Python's random module inside DataLoader workers.

    PyTorch assigns each worker its own torch seed from the
    DataLoader generator.
    """

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(
        worker_seed
    )


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

def validate_linear_probe_model(
    model: HulkHandDinoV2,
) -> None:
    """
    V5 Phase 1 is intentionally a strict linear probe.

    The entire DINOv2 backbone must remain frozen.

    Only:

        classifier.weight
        classifier.bias

    may be trainable.
    """

    backbone_trainable = [
        name
        for name, parameter
        in model.backbone.named_parameters()
        if parameter.requires_grad
    ]

    if backbone_trainable:
        raise RuntimeError(
            "DINOv2 backbone contains trainable "
            "parameters:\n"
            + "\n".join(
                backbone_trainable
            )
        )

    trainable_names = {
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    expected = {
        "classifier.weight",
        "classifier.bias",
    }

    if trainable_names != expected:
        raise RuntimeError(
            "Unexpected trainable parameters.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(trainable_names)}"
        )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int, int]:

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


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: HulkHandDinoV2,
) -> SGD:

    return SGD(
        model.classifier.parameters(),
        lr=CLASSIFIER_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: SGD,
) -> CosineAnnealingLR:

    return CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=0.0,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_train_loader(
    combined_dataset,
    batch_sampler,
    epoch: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:

    generator = torch.Generator()

    generator.manual_seed(
        seed + epoch
    )

    # Epoch 1 -> sampler epoch 0.
    batch_sampler.set_epoch(
        epoch - 1
    )

    return DataLoader(
        combined_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def build_validation_loader(
    dataset,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------

def require_finite_tensor(
    tensor: torch.Tensor,
    name: str,
    epoch: int,
    batch_index: int,
) -> None:

    if not torch.isfinite(
        tensor
    ).all():

        raise FloatingPointError(
            f"Non-finite values detected in {name} "
            f"during epoch {epoch}, batch {batch_index}."
        )


def require_finite_loss(
    loss: torch.Tensor,
    epoch: int,
    batch_index: int,
    phase: str,
) -> None:

    if not torch.isfinite(
        loss
    ):

        raise FloatingPointError(
            f"Non-finite {phase} loss during "
            f"epoch {epoch}, batch {batch_index}: "
            f"{loss.detach().item()}"
        )


def calculate_gradient_norm(
    parameters: list[torch.Tensor],
) -> torch.Tensor:

    squared_norms = []

    for parameter in parameters:

        if parameter.grad is None:
            continue

        squared_norms.append(
            torch.sum(
                parameter.grad.detach()
                * parameter.grad.detach()
            )
        )

    if not squared_norms:

        raise RuntimeError(
            "No gradients were available for "
            "gradient-norm calculation."
        )

    total_squared = torch.stack(
        squared_norms
    ).sum()

    return torch.sqrt(
        total_squared
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: HulkHandDinoV2,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: SGD,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:

    model.train()

    # HulkHandDinoV2.train() keeps the frozen backbone
    # permanently in eval mode.
    if model.backbone.training:
        raise RuntimeError(
            "Frozen DINOv2 backbone entered training mode."
        )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    progress = tqdm(
        loader,
        total=BATCHES_PER_EPOCH,
        desc=f"Epoch {epoch:02d} train",
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress,
        start=1,
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        # ------------------------------------------------------
        # FP32 forward
        # ------------------------------------------------------

        outputs = model(
            images
        )

        logits = outputs[
            "logits"
        ]

        embedding = outputs[
            "embedding"
        ]

        require_finite_tensor(
            logits,
            name="training logits",
            epoch=epoch,
            batch_index=batch_index,
        )

        require_finite_tensor(
            embedding,
            name="training embedding",
            epoch=epoch,
            batch_index=batch_index,
        )

        loss = criterion(
            logits,
            labels,
        )

        require_finite_loss(
            loss,
            epoch=epoch,
            batch_index=batch_index,
            phase="training",
        )

        # ------------------------------------------------------
        # FP32 backward
        # ------------------------------------------------------

        loss.backward()

        # Check every trainable gradient.
        for name, parameter in (
            model.named_parameters()
        ):

            if not parameter.requires_grad:
                continue

            if parameter.grad is None:
                raise RuntimeError(
                    f"Trainable parameter '{name}' received "
                    f"no gradient during epoch {epoch}, "
                    f"batch {batch_index}."
                )

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise FloatingPointError(
                    f"Non-finite gradient in '{name}' during "
                    f"epoch {epoch}, batch {batch_index}."
                )

        gradient_norm = (
            calculate_gradient_norm(
                trainable_parameters
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):

            raise FloatingPointError(
                "Non-finite gradient norm during "
                f"epoch {epoch}, batch {batch_index}."
            )

        # No gradient clipping for the V5 linear probe.
        optimizer.step()

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

        running_loss = (
            total_loss
            / total_samples
        )

        running_accuracy = (
            total_correct
            / total_samples
        )

        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_accuracy * 100:.2f}%",
        )

    if total_samples == 0:
        raise RuntimeError(
            "Training loader produced no samples."
        )

    return (
        total_loss
        / total_samples,
        total_correct
        / total_samples,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: HulkHandDinoV2,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    description: str,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress,
        start=1,
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        logits = outputs[
            "logits"
        ]

        embedding = outputs[
            "embedding"
        ]

        require_finite_tensor(
            logits,
            name=f"{description} logits",
            epoch=epoch,
            batch_index=batch_index,
        )

        require_finite_tensor(
            embedding,
            name=f"{description} embedding",
            epoch=epoch,
            batch_index=batch_index,
        )

        loss = criterion(
            logits,
            labels,
        )

        require_finite_loss(
            loss,
            epoch=epoch,
            batch_index=batch_index,
            phase=description,
        )

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise RuntimeError(
            f"{description} dataset is empty."
        )

    return (
        total_loss
        / total_samples,
        total_correct
        / total_samples,
    )


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------

def capture_rng_state() -> dict[str, Any]:

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        state[
            "cuda"
        ] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
    state: dict[str, Any] | None,
) -> None:

    if not state:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        torch.cuda.is_available()
        and "cuda" in state
    ):

        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def build_checkpoint(
    model: HulkHandDinoV2,
    optimizer: SGD,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_new_val_accuracy: float,
    early_stopping_counter: int,
    classes: list[str],
    class_to_idx: dict[str, int],
    train_loss: float,
    train_accuracy: float,
    old_val_loss: float,
    old_val_accuracy: float,
    new_val_loss: float,
    new_val_accuracy: float,
) -> dict[str, Any]:

    return {
        "epoch": epoch,

        "model_id": MODEL_ID,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "best_new_val_accuracy": (
            best_new_val_accuracy
        ),

        "early_stopping_counter": (
            early_stopping_counter
        ),

        "classes": classes,

        "class_to_idx": (
            class_to_idx
        ),

        "num_classes": (
            len(classes)
        ),

        "metrics": {
            "train_loss": (
                train_loss
            ),
            "train_accuracy": (
                train_accuracy
            ),
            "old_val_loss": (
                old_val_loss
            ),
            "old_val_accuracy": (
                old_val_accuracy
            ),
            "new_val_loss": (
                new_val_loss
            ),
            "new_val_accuracy": (
                new_val_accuracy
            ),
        },

        "training_config": {
            "model_id": MODEL_ID,
            "backbone": DINOV2_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "precision": PRECISION,
            "training_mode": "linear_probe",
            "backbone_frozen": True,
            "classifier_lr": CLASSIFIER_LR,
            "optimizer": "sgd",
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "gradient_clipping": False,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_batch_size": (
                VALIDATION_BATCH_SIZE
            ),
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "samples_per_epoch": (
                SAMPLES_PER_EPOCH
            ),
            "source_split": SOURCE_SPLIT,
        },

        "rng_state": (
            capture_rng_state()
        ),
    }


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: Path,
) -> None:
    """
    Atomic checkpoint write.

    The previous checkpoint remains intact if the new
    write is interrupted before replacement.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_name(
            path.name + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        path
    )


def load_checkpoint_file(
    path: Path,
) -> dict[str, Any]:

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}"
        )

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Invalid training checkpoint."
        )

    required_keys = {
        "epoch",
        "model_id",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_new_val_accuracy",
        "early_stopping_counter",
        "classes",
        "class_to_idx",
    }

    missing = (
        required_keys
        - set(
            checkpoint.keys()
        )
    )

    if missing:

        raise RuntimeError(
            "Resume checkpoint is missing keys: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    return checkpoint


def move_optimizer_state_to_device(
    optimizer: SGD,
    device: torch.device,
) -> None:

    for state in (
        optimizer.state.values()
    ):

        for key, value in (
            state.items()
        ):

            if isinstance(
                value,
                torch.Tensor,
            ):

                state[
                    key
                ] = value.to(
                    device
                )


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------

def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    classes: list[str],
    class_to_idx: dict[str, int],
) -> None:

    saved_model_id = (
        checkpoint.get(
            "model_id"
        )
    )

    if saved_model_id != MODEL_ID:

        raise RuntimeError(
            "Resume checkpoint belongs to a "
            "different model configuration.\n"
            f"Expected: {MODEL_ID}\n"
            f"Actual:   {saved_model_id}"
        )

    saved_classes = (
        checkpoint.get(
            "classes"
        )
    )

    saved_mapping = (
        checkpoint.get(
            "class_to_idx"
        )
    )

    if saved_classes != classes:

        raise RuntimeError(
            "Dataset classes differ from the "
            "resume checkpoint.\n"
            f"Checkpoint: {saved_classes}\n"
            f"Current:    {classes}"
        )

    if saved_mapping != class_to_idx:

        raise RuntimeError(
            "Class mapping differs from the "
            "resume checkpoint."
        )

    saved_num_classes = (
        checkpoint.get(
            "num_classes"
        )
    )

    if (
        saved_num_classes is not None
        and int(
            saved_num_classes
        ) != len(
            classes
        )
    ):

        raise RuntimeError(
            "Number of classes differs from "
            "the resume checkpoint."
        )

    training_config = (
        checkpoint.get(
            "training_config",
            {},
        )
    )

    saved_precision = (
        training_config.get(
            "precision"
        )
    )

    if (
        saved_precision is not None
        and saved_precision
        != PRECISION
    ):

        raise RuntimeError(
            "Resume checkpoint was created with "
            f"precision '{saved_precision}', but "
            f"this pipeline requires '{PRECISION}'."
        )

    saved_training_mode = (
        training_config.get(
            "training_mode"
        )
    )

    if (
        saved_training_mode
        is not None
        and saved_training_mode
        != "linear_probe"
    ):

        raise RuntimeError(
            "Resume checkpoint is not a "
            "DINOv2 linear-probe checkpoint."
        )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_training_header(
    device: torch.device,
    classes: list[str],
    output_dir: Path,
    datasets: dict,
    model: HulkHandDinoV2,
) -> None:

    (
        total_parameters,
        frozen_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    print()
    print(
        "hulk-hand DINOv2 training"
    )
    print(
        "========================="
    )
    print()

    print(
        f"PyTorch:             "
        f"{torch.__version__}"
    )

    print(
        f"Device:              "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            "GPU:                 "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA:                "
            f"{torch.version.cuda}"
        )

    print(
        f"Precision:           "
        f"{PRECISION.upper()}"
    )

    print(
        f"Backbone:            "
        f"{DINOV2_MODEL}"
    )

    print(
        "Training mode:       "
        "linear probe"
    )

    print(
        "Backbone frozen:     yes"
    )

    print(
        "Attention:           "
        "Opset13 classic"
    )

    print(
        f"Embedding dim:       "
        f"{EMBEDDING_DIM}"
    )

    print()

    print(
        f"Total parameters:    "
        f"{total_parameters:,}"
    )

    print(
        f"Frozen parameters:   "
        f"{frozen_parameters:,}"
    )

    print(
        f"Trainable parameters:"
        f" {trainable_parameters:,}"
    )

    print()

    print(
        f"Classes:             "
        f"{len(classes)}"
    )

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {index}: {class_name}"
        )

    print()

    print(
        f"OLD train:           "
        f"{len(datasets['old_train'])}"
    )

    print(
        f"OLD val:             "
        f"{len(datasets['old_val'])}"
    )

    print(
        f"NEW train:           "
        f"{len(datasets['new_train'])}"
    )

    print(
        f"NEW val:             "
        f"{len(datasets['new_val'])}"
    )

    print(
        f"NEW test:            "
        f"{len(datasets['new_test'])}"
    )

    print()

    print(
        f"Batch size:          "
        f"{BATCH_SIZE}"
    )

    print(
        f"Batches / epoch:     "
        f"{BATCHES_PER_EPOCH}"
    )

    print(
        f"Samples / epoch:     "
        f"{SAMPLES_PER_EPOCH}"
    )

    print(
        f"OLD / NEW:           "
        f"{SOURCE_SPLIT}"
    )

    print()

    print(
        f"Optimizer:           SGD"
    )

    print(
        f"Classifier LR:       "
        f"{CLASSIFIER_LR}"
    )

    print(
        f"Momentum:            "
        f"{MOMENTUM}"
    )

    print(
        f"Weight decay:        "
        f"{WEIGHT_DECAY}"
    )

    print(
        "Warm-up epochs:      0"
    )

    print(
        "Scheduler:           cosine"
    )

    print(
        "Gradient clipping:   disabled"
    )

    print()

    print(
        f"Maximum epochs:      "
        f"{MAX_EPOCHS}"
    )

    print(
        "Early stop patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Output:              "
        f"{output_dir}"
    )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the hulk-hand DINOv2 ViT-S/14 "
            "linear classifier in FP32."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of the processed dataset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for best.pt and last.pt."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume training from a V5 last.pt."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "DataLoader worker processes. Default: 4"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed. Default: 42"
        ),
    )

    parser.add_argument(
        "--horizontal-flip",
        action="store_true",
        help=(
            "Enable horizontal-flip augmentation. "
            "Only use when left/right hand orientation "
            "has identical semantic meaning."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers must be >= 0."
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not data_dir.is_dir():

        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "best.pt"
    )

    last_path = (
        output_dir
        / "last.pt"
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = (
        device.type
        == "cuda"
    )

    set_seed(
        args.seed
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    (
        datasets,
        classes,
        class_to_idx,
    ) = build_datasets(
        data_dir=data_dir,
        image_size=224,
        horizontal_flip=(
            args.horizontal_flip
        ),
        old_val_fraction=0.10,
        seed=args.seed,
    )

    old_train = datasets[
        "old_train"
    ]

    new_train = datasets[
        "new_train"
    ]

    old_val = datasets[
        "old_val"
    ]

    new_val = datasets[
        "new_val"
    ]

    # NEW test is intentionally not used here.
    # Sessions W-Z remain untouched until final evaluation.

    if len(
        old_train
    ) == 0:

        raise RuntimeError(
            "OLD training dataset is empty."
        )

    if len(
        new_train
    ) == 0:

        raise RuntimeError(
            "NEW training dataset is empty."
        )

    if len(
        old_val
    ) == 0:

        raise RuntimeError(
            "OLD validation dataset is empty."
        )

    if len(
        new_val
    ) == 0:

        raise RuntimeError(
            "NEW validation dataset is empty."
        )

    combined_train, batch_sampler = (
        build_training_dataset_and_sampler(
            old_train=old_train,
            new_train=new_train,
            seed=args.seed,
        )
    )

    old_val_loader = (
        build_validation_loader(
            old_val,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    )

    new_val_loader = (
        build_validation_loader(
            new_val,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    )

    # ------------------------------------------------------------------
    # Resume checkpoint
    # ------------------------------------------------------------------

    resume_checkpoint = None

    if args.resume is not None:

        resume_checkpoint = (
            load_checkpoint_file(
                args.resume
            )
        )

        validate_resume_checkpoint(
            resume_checkpoint,
            classes,
            class_to_idx,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    if resume_checkpoint is None:

        print()
        print(
            "Loading pretrained DINOv2 "
            "ViT-S/14 backbone..."
        )
        print()

        model = build_model(
            num_classes=len(
                classes
            ),
            pretrained=True,
        )

    else:

        # The full pretrained backbone is contained in
        # last.pt, so pretrained weights do not need to
        # be downloaded again.
        model = build_model(
            num_classes=len(
                classes
            ),
            pretrained=False,
        )

    validate_linear_probe_model(
        model
    )

    model.to(
        device
    )

    print_training_header(
        device=device,
        classes=classes,
        output_dir=output_dir,
        datasets=datasets,
        model=model,
    )

    # ------------------------------------------------------------------
    # Training components
    # ------------------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    scheduler = build_scheduler(
        optimizer
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    start_epoch = 1

    best_new_val_accuracy = (
        float("-inf")
    )

    early_stopping_counter = 0

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    if resume_checkpoint is not None:

        model.load_state_dict(
            resume_checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            resume_checkpoint[
                "optimizer_state_dict"
            ]
        )

        move_optimizer_state_to_device(
            optimizer,
            device,
        )

        scheduler.load_state_dict(
            resume_checkpoint[
                "scheduler_state_dict"
            ]
        )

        previous_epoch = int(
            resume_checkpoint[
                "epoch"
            ]
        )

        start_epoch = (
            previous_epoch
            + 1
        )

        best_new_val_accuracy = float(
            resume_checkpoint[
                "best_new_val_accuracy"
            ]
        )

        early_stopping_counter = int(
            resume_checkpoint[
                "early_stopping_counter"
            ]
        )

        restore_rng_state(
            resume_checkpoint.get(
                "rng_state"
            )
        )

        validate_linear_probe_model(
            model
        )

        print(
            f"Resuming after epoch "
            f"{previous_epoch}."
        )

        print(
            "Best NEW validation accuracy: "
            f"{best_new_val_accuracy * 100:.2f}%"
        )

        print(
            "Early stopping counter: "
            f"{early_stopping_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        print()

    if start_epoch > MAX_EPOCHS:

        print(
            "Training has already reached "
            f"{MAX_EPOCHS} epochs."
        )

        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        # Deterministic epoch-specific randomness.
        set_seed(
            args.seed
            + epoch
        )

        train_loader = (
            build_train_loader(
                combined_dataset=(
                    combined_train
                ),
                batch_sampler=(
                    batch_sampler
                ),
                epoch=epoch,
                num_workers=(
                    args.num_workers
                ),
                pin_memory=(
                    pin_memory
                ),
                seed=args.seed,
            )
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        (
            old_val_loss,
            old_val_accuracy,
        ) = evaluate(
            model=model,
            loader=old_val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            description=(
                f"Epoch {epoch:02d} OLD val"
            ),
        )

        (
            new_val_loss,
            new_val_accuracy,
        ) = evaluate(
            model=model,
            loader=new_val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            description=(
                f"Epoch {epoch:02d} NEW val"
            ),
        )

        # --------------------------------------------------------------
        # Best model / early stopping
        # --------------------------------------------------------------

        improved = (
            new_val_accuracy
            > best_new_val_accuracy
        )

        if improved:

            best_new_val_accuracy = (
                new_val_accuracy
            )

            early_stopping_counter = 0

        else:

            early_stopping_counter += 1

        # Scheduler advances only after this epoch.
        scheduler.step()

        # --------------------------------------------------------------
        # Console summary
        # --------------------------------------------------------------

        print(
            f"Epoch {epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        print(
            f"  Train loss:       "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train accuracy:   "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"  OLD val loss:     "
            f"{old_val_loss:.4f}"
        )

        print(
            f"  OLD val accuracy: "
            f"{old_val_accuracy * 100:.2f}%"
        )

        print(
            f"  NEW val loss:     "
            f"{new_val_loss:.4f}"
        )

        print(
            f"  NEW val accuracy: "
            f"{new_val_accuracy * 100:.2f}%"
        )

        print(
            f"  Classifier LR:    "
            f"{current_lr:.8f}"
        )

        print(
            "  Early stopping:   "
            f"{early_stopping_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        if improved:

            print(
                "  BEST:             yes"
            )

        print()

        # --------------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------------

        checkpoint = (
            build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_new_val_accuracy=(
                    best_new_val_accuracy
                ),
                early_stopping_counter=(
                    early_stopping_counter
                ),
                classes=classes,
                class_to_idx=(
                    class_to_idx
                ),
                train_loss=(
                    train_loss
                ),
                train_accuracy=(
                    train_accuracy
                ),
                old_val_loss=(
                    old_val_loss
                ),
                old_val_accuracy=(
                    old_val_accuracy
                ),
                new_val_loss=(
                    new_val_loss
                ),
                new_val_accuracy=(
                    new_val_accuracy
                ),
            )
        )

        # Always save complete latest state.
        save_checkpoint(
            checkpoint,
            last_path,
        )

        # Save best checkpoint only when NEW
        # validation accuracy improves.
        if improved:

            save_checkpoint(
                checkpoint,
                best_path,
            )

            print(
                f"Saved best checkpoint: "
                f"{best_path}"
            )

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------

        if (
            early_stopping_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping triggered."
            )

            print(
                "Best NEW validation accuracy: "
                f"{best_new_val_accuracy * 100:.2f}%"
            )

            break

    print()
    print(
        "Training finished."
    )

    print(
        f"Best checkpoint: "
        f"{best_path}"
    )

    print(
        f"Last checkpoint: "
        f"{last_path}"
    )

    print()


if __name__ == "__main__":
    main()