#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from model import (
    DINOV2_MODEL,
    EMBEDDING_DIM,
    HulkHandDinoV2,
    NUM_TRAINABLE_BLOCKS,
    build_model,
    validate_trainable_parameters,
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

MODEL_ID = "dinov2_vits14_v9_layerwise_decay_last8_opset13"

MAX_EPOCHS = 40

WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.10

BACKBONE_LR = 5e-6
BACKBONE_LR_DECAY = 0.85
CLASSIFIER_LR = 2e-4

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

GRADIENT_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 8

VALIDATION_BATCH_SIZE = 32

PRECISION = "fp32"

SOURCE_SPLIT = "25% OLD / 75% NEW"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: HulkHandDinoV2,
) -> AdamW:

    parameter_groups = []

    trainable_block_indices = list(
        model.trainable_block_indices
    )

    if not trainable_block_indices:
        raise RuntimeError(
            "No trainable DINOv2 blocks found."
        )

    last_block_position = len(
        trainable_block_indices
    ) - 1

    for position, block_index in enumerate(
        trainable_block_indices
    ):

        block_parameters = [
            parameter
            for parameter in model.backbone.blocks[
                block_index
            ].parameters()
            if parameter.requires_grad
        ]

        if not block_parameters:
            continue

        lr = BACKBONE_LR * (
            BACKBONE_LR_DECAY
            ** (last_block_position - position)
        )

        parameter_groups.append(
            {
                "params": block_parameters,
                "lr": lr,
                "name": f"backbone_block_{block_index}",
            }
        )

    norm_parameters = [
        parameter
        for parameter in model.backbone.norm.parameters()
        if parameter.requires_grad
    ]

    if norm_parameters:
        parameter_groups.append(
            {
                "params": norm_parameters,
                "lr": BACKBONE_LR,
                "name": "backbone_norm",
            }
        )

    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    if not parameter_groups:
        raise RuntimeError(
            "No trainable DINOv2 parameters found."
        )

    if not classifier_parameters:
        raise RuntimeError(
            "No trainable classifier parameters found."
        )

    parameter_groups.append(
        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR,
            "name": "classifier",
        }
    )

    return AdamW(
        parameter_groups,
        weight_decay=WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: AdamW,
) -> SequentialLR:

    warmup = LinearLR(
        optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS - WARMUP_EPOCHS,
        eta_min=0.0,
    )

    return SequentialLR(
        optimizer,
        schedulers=[
            warmup,
            cosine,
        ],
        milestones=[
            WARMUP_EPOCHS,
        ],
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
# Numerical checks
# ---------------------------------------------------------------------------

def require_finite_tensor(
    tensor: torch.Tensor,
    name: str,
    epoch: int,
    batch_index: int,
) -> None:

    if not torch.isfinite(tensor).all():

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

    if not torch.isfinite(loss):

        raise FloatingPointError(
            f"Non-finite {phase} loss during "
            f"epoch {epoch}, batch {batch_index}: "
            f"{loss.detach().item()}"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: HulkHandDinoV2,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:

    model.train()

    for block_index in range(
        model.trainable_block_indices[0]
    ):

        if model.backbone.blocks[
            block_index
        ].training:

            raise RuntimeError(
                f"Frozen DINOv2 block "
                f"{block_index} entered training mode."
            )

    for block_index in model.trainable_block_indices:

        if not model.backbone.blocks[
            block_index
        ].training:

            raise RuntimeError(
                f"Trainable DINOv2 block "
                f"{block_index} is not in training mode."
            )

    if not model.backbone.norm.training:

        raise RuntimeError(
            "Final DINOv2 norm is not in training mode."
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

        outputs = model(images)

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

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=GRADIENT_CLIP_NORM,
        )

        if not torch.isfinite(
            torch.as_tensor(grad_norm)
        ):

            raise FloatingPointError(
                "Non-finite gradient norm during "
                f"epoch {epoch}, batch {batch_index}: "
                f"{float(grad_norm)}"
            )

        for name, parameter in model.named_parameters():

            if parameter.requires_grad:

                if parameter.grad is None:

                    raise RuntimeError(
                        f"Trainable parameter '{name}' "
                        f"received no gradient during "
                        f"epoch {epoch}, batch {batch_index}."
                    )

                if not torch.isfinite(
                    parameter.grad
                ).all():

                    raise FloatingPointError(
                        f"Non-finite gradient in '{name}' "
                        f"during epoch {epoch}, "
                        f"batch {batch_index}."
                    )

            elif parameter.grad is not None:

                raise RuntimeError(
                    f"Frozen parameter '{name}' "
                    "received a gradient."
                )

        optimizer.step()

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        batch_size = labels.size(0)

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

        total_samples += batch_size

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
        total_loss / total_samples,
        total_correct / total_samples,
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

        outputs = model(images)

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

        batch_size = labels.size(0)

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

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            f"{description} dataset is empty."
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
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
    optimizer: AdamW,
    scheduler: SequentialLR,
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

        "class_to_idx": class_to_idx,

        "num_classes": len(classes),

        "metrics": {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "old_val_loss": old_val_loss,
            "old_val_accuracy": old_val_accuracy,
            "new_val_loss": new_val_loss,
            "new_val_accuracy": new_val_accuracy,
        },

        "training_config": {
            "model_id": MODEL_ID,
            "backbone": DINOV2_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "precision": PRECISION,

            "training_mode": (
                "last_8_blocks_layerwise_decay"
            ),

            "num_trainable_blocks": len(
                model.trainable_block_indices
            ),

            "trainable_blocks": list(
                model.trainable_block_indices
            ),

            "final_norm_trainable": True,

            "backbone_lr": BACKBONE_LR,
            "backbone_lr_decay": BACKBONE_LR_DECAY,
            "classifier_lr": CLASSIFIER_LR,
            "label_smoothing": LABEL_SMOOTHING,

            "optimizer": "adamw",
            "weight_decay": WEIGHT_DECAY,

            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_start_factor": (
                WARMUP_START_FACTOR
            ),

            "scheduler": "cosine",

            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),

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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
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
        - set(checkpoint.keys())
    )

    if missing:

        raise RuntimeError(
            "Resume checkpoint is missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    return checkpoint


def move_optimizer_state_to_device(
    optimizer: AdamW,
    device: torch.device,
) -> None:

    for state in optimizer.state.values():

        for key, value in state.items():

            if isinstance(
                value,
                torch.Tensor,
            ):

                state[key] = value.to(
                    device
                )


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------

def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    classes: list[str],
    class_to_idx: dict[str, int],
    num_trainable_blocks: int,
) -> None:

    saved_model_id = checkpoint.get(
        "model_id"
    )

    if saved_model_id != MODEL_ID:

        raise RuntimeError(
            "Resume checkpoint belongs to a "
            "different model configuration.\n"
            f"Expected: {MODEL_ID}\n"
            f"Actual:   {saved_model_id}"
        )

    saved_classes = checkpoint.get(
        "classes"
    )

    saved_mapping = checkpoint.get(
        "class_to_idx"
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

    saved_num_classes = checkpoint.get(
        "num_classes"
    )

    if (
        saved_num_classes is not None
        and int(saved_num_classes)
        != len(classes)
    ):

        raise RuntimeError(
            "Number of classes differs from "
            "the resume checkpoint."
        )

    training_config = checkpoint.get(
        "training_config",
        {},
    )

    saved_precision = training_config.get(
        "precision"
    )

    if (
        saved_precision is not None
        and saved_precision != PRECISION
    ):

        raise RuntimeError(
            "Resume checkpoint was created "
            f"with precision '{saved_precision}', "
            f"but this pipeline requires "
            f"'{PRECISION}'."
        )

    saved_mode = training_config.get(
        "training_mode"
    )

    if (
        saved_mode is not None
        and saved_mode
        != "last_8_blocks_layerwise_decay"
    ):

        raise RuntimeError(
            "Resume checkpoint was created "
            "with a different training mode."
        )

    saved_trainable_blocks = training_config.get(
        "num_trainable_blocks"
    )

    if (
        saved_trainable_blocks is not None
        and int(saved_trainable_blocks)
        != num_trainable_blocks
    ):

        raise RuntimeError(
            "Resume checkpoint was created with a different number of trainable blocks."
        )


# ---------------------------------------------------------------------------
# Parameter statistics
# ---------------------------------------------------------------------------

def count_parameters(
    model: nn.Module,
) -> tuple[int, int, int]:

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


# ---------------------------------------------------------------------------
# Header
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
        "hulk-hand DINOv2 V9 training"
    )
    print(
        "============================"
    )
    print()

    print(
        f"PyTorch:             {torch.__version__}"
    )

    print(
        f"Device:              {device}"
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
        "layerwise decay on last 8 blocks + norm"
    )

    print(
        "Trainable blocks:    "
        + ", ".join(
            str(index)
            for index
            in model.trainable_block_indices
        )
    )

    print(
        "Final norm:          trainable"
    )

    print(
        "Attention:           Opset13 classic"
    )

    print(
        f"Embedding dim:       {EMBEDDING_DIM}"
    )

    print()

    print(
        f"Total parameters:    {total_parameters:,}"
    )

    print(
        f"Frozen parameters:   {frozen_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    print()

    print(
        f"Classes:             {len(classes)}"
    )

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {index}: {class_name}"
        )

    print()

    print(
        f"OLD train:           {len(datasets['old_train'])}"
    )

    print(
        f"OLD val:             {len(datasets['old_val'])}"
    )

    print(
        f"NEW train:           {len(datasets['new_train'])}"
    )

    print(
        f"NEW val:             {len(datasets['new_val'])}"
    )

    print(
        f"NEW test:            {len(datasets['new_test'])}"
    )

    print()

    print(
        f"Batch size:          {BATCH_SIZE}"
    )

    print(
        f"Batches / epoch:     {BATCHES_PER_EPOCH}"
    )

    print(
        f"Samples / epoch:     {SAMPLES_PER_EPOCH}"
    )

    print(
        f"OLD / NEW:           {SOURCE_SPLIT}"
    )

    print()

    print(
        "Optimizer:           AdamW"
    )

    print(
        f"LR backbone:         {BACKBONE_LR}"
    )

    print(
        f"LR classifier:       {CLASSIFIER_LR}"
    )

    print(
        f"Weight decay:        {WEIGHT_DECAY}"
    )

    print(
        f"Warm-up epochs:      {WARMUP_EPOCHS}"
    )

    print(
        "Scheduler:           cosine"
    )

    print(
        f"Gradient clip:       {GRADIENT_CLIP_NORM}"
    )

    print()

    print(
        f"Maximum epochs:      {MAX_EPOCHS}"
    )

    print(
        "Early stop patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Output:              {output_dir}"
    )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "V9 layerwise-decay fine-tuning of the "
            "last eight DINOv2 ViT-S/14 transformer "
            "blocks for hulk-hand."
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
            "Resume training from a V9 last.pt."
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
            "Enable horizontal-flip augmentation."
        ),
    )

    args = parser.parse_args()

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
            f"Dataset directory not found: "
            f"{data_dir}"
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
        device.type == "cuda"
    )

    set_seed(
        args.seed
    )

    # ------------------------------------------------------------------
    # Dataset
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

    # W-Z remain untouched.

    if len(old_train) == 0:
        raise RuntimeError(
            "OLD training dataset is empty."
        )

    if len(new_train) == 0:
        raise RuntimeError(
            "NEW training dataset is empty."
        )

    if len(old_val) == 0:
        raise RuntimeError(
            "OLD validation dataset is empty."
        )

    if len(new_val) == 0:
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

        resume_checkpoint = load_checkpoint_file(
            args.resume
        )

        validate_resume_checkpoint(
            resume_checkpoint,
            classes,
            class_to_idx,
            NUM_TRAINABLE_BLOCKS,
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
            num_classes=len(classes),
            pretrained=True,
            num_trainable_blocks=NUM_TRAINABLE_BLOCKS,
        )

    else:

        model = build_model(
            num_classes=len(classes),
            pretrained=False,
            num_trainable_blocks=NUM_TRAINABLE_BLOCKS,
        )

    validate_trainable_parameters(
        model
    )

    model.to(
        device
    )

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    scheduler = build_scheduler(
        optimizer
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
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
            previous_epoch + 1
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

        validate_trainable_parameters(
            model
        )

        print()
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

    print_training_header(
        device=device,
        classes=classes,
        output_dir=output_dir,
        datasets=datasets,
        model=model,
    )

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

        set_seed(
            args.seed + epoch
        )

        train_loader = (
            build_train_loader(
                combined_dataset=combined_train,
                batch_sampler=batch_sampler,
                epoch=epoch,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                seed=args.seed,
            )
        )

        backbone_lrs = [
            group["lr"]
            for group in optimizer.param_groups
            if str(group.get("name", "")).startswith(
                "backbone_block_"
            )
            or group.get("name") == "backbone_norm"
        ]

        current_backbone_lr_min = min(
            backbone_lrs
        )

        current_backbone_lr_max = max(
            backbone_lrs
        )

        current_classifier_lr = next(
            group["lr"]
            for group in optimizer.param_groups
            if group.get("name") == "classifier"
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
        # Best / early stopping
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

        scheduler.step()

        # --------------------------------------------------------------
        # Summary
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
            f"  LR backbone:      "
            f"{current_backbone_lr_min:.8f}.."
            f"{current_backbone_lr_max:.8f}"
        )

        print(
            f"  LR classifier:    "
            f"{current_classifier_lr:.8f}"
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

        checkpoint = build_checkpoint(
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
            class_to_idx=class_to_idx,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            old_val_loss=old_val_loss,
            old_val_accuracy=(
                old_val_accuracy
            ),
            new_val_loss=new_val_loss,
            new_val_accuracy=(
                new_val_accuracy
            ),
        )

        save_checkpoint(
            checkpoint,
            last_path,
        )

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
        f"Best checkpoint: {best_path}"
    )

    print(
        f"Last checkpoint: {last_path}"
    )

    print()


if __name__ == "__main__":
    main()
