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
from model import HulkHandResNet18, build_model
from sampler import (
    BATCH_SIZE,
    BATCHES_PER_EPOCH,
    SAMPLES_PER_EPOCH,
    build_training_dataset_and_sampler,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MAX_EPOCHS = 30

WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.10

LAYER4_LR = 3e-5
CLASSIFIER_LR = 1.5e-4

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 5

VALIDATION_BATCH_SIZE = 128

PRECISION = "fp32"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Seed Python's random module inside DataLoader workers.

    PyTorch assigns each worker its own torch seed from the
    DataLoader generator.
    """

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: HulkHandResNet18,
) -> AdamW:

    return AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": LAYER4_LR,
                "name": "layer4",
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LR,
                "name": "classifier",
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# Learning-rate scheduler
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
    generator.manual_seed(seed + epoch)

    # Epoch 1 -> sampler epoch 0.
    batch_sampler.set_epoch(epoch - 1)

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
    model: HulkHandResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

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
        logits = outputs["logits"]

        require_finite_tensor(
            logits,
            name="training logits",
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

        # Check individual trainable gradients as well.
        for name, parameter in model.named_parameters():

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
            dim=1,
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
    model: HulkHandResNet18,
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
        logits = outputs["logits"]

        require_finite_tensor(
            logits,
            name=f"{description} logits",
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
            dim=1,
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
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

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
    model: HulkHandResNet18,
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
            "precision": PRECISION,
            "max_epochs": MAX_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_start_factor": WARMUP_START_FACTOR,
            "layer4_lr": LAYER4_LR,
            "classifier_lr": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "batch_size": BATCH_SIZE,
            "validation_batch_size": VALIDATION_BATCH_SIZE,
            "batches_per_epoch": BATCHES_PER_EPOCH,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
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

    The previous checkpoint remains intact if the new write
    is interrupted before replacement.
    """

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
            + ", ".join(sorted(missing))
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
) -> None:

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
        and int(saved_num_classes) != len(classes)
    ):
        raise RuntimeError(
            "Number of classes differs from the "
            "resume checkpoint."
        )

    training_config = checkpoint.get(
        "training_config",
        {},
    )

    saved_precision = training_config.get(
        "precision"
    )

    # Reject explicitly non-FP32 checkpoints.
    # Older checkpoints without a precision field are allowed.
    if (
        saved_precision is not None
        and saved_precision != PRECISION
    ):
        raise RuntimeError(
            "Resume checkpoint was created with "
            f"precision '{saved_precision}', but this "
            f"training pipeline requires '{PRECISION}'."
        )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_training_header(
    device: torch.device,
    classes: list[str],
    output_dir: Path,
    datasets: dict,
) -> None:

    print()
    print("hulk-hand training")
    print("==================")
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
        f"Precision:           {PRECISION.upper()}"
    )

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
        "OLD / NEW:           50% / 50%"
    )

    print(
        f"Maximum epochs:      {MAX_EPOCHS}"
    )

    print(
        f"Warm-up epochs:      {WARMUP_EPOCHS}"
    )

    print(
        f"LR layer4:           {LAYER4_LR}"
    )

    print(
        f"LR classifier:       {CLASSIFIER_LR}"
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
            "Fine-tune the hulk-hand HaGRID "
            "ResNet-18 model in FP32."
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
        "--hagrid-checkpoint",
        type=Path,
        default=None,
        help=(
            "Pretrained HaGRID ResNet-18 checkpoint. "
            "Required for a fresh training run."
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
            "Resume training from a previous last.pt."
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

    if (
        args.resume is None
        and args.hagrid_checkpoint is None
    ):
        raise ValueError(
            "--hagrid-checkpoint is required "
            "for a fresh training run."
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

    hagrid_checkpoint = None

    if args.hagrid_checkpoint is not None:

        hagrid_checkpoint = (
            args.hagrid_checkpoint
            .expanduser()
            .resolve()
        )

        if (
            args.resume is None
            and not hagrid_checkpoint.is_file()
        ):
            raise FileNotFoundError(
                "HaGRID checkpoint not found: "
                f"{hagrid_checkpoint}"
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
    # Datasets
    # ------------------------------------------------------------------

    datasets, classes, class_to_idx = (
        build_datasets(
            data_dir=data_dir,
            image_size=224,
            horizontal_flip=(
                args.horizontal_flip
            ),
            old_val_fraction=0.10,
            seed=args.seed,
        )
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

    print_training_header(
        device=device,
        classes=classes,
        output_dir=output_dir,
        datasets=datasets,
    )

    # ------------------------------------------------------------------
    # Model / resume checkpoint
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

        model = HulkHandResNet18(
            num_classes=len(classes)
        )

    else:

        assert hagrid_checkpoint is not None

        model = build_model(
            num_classes=len(classes),
            checkpoint_path=hagrid_checkpoint,
        )

    model.to(
        device
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

    criterion = nn.CrossEntropyLoss()

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

        current_lrs = [
            group["lr"]
            for group in optimizer.param_groups
        ]

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        old_val_loss, old_val_accuracy = (
            evaluate(
                model=model,
                loader=old_val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                description=(
                    f"Epoch {epoch:02d} OLD val"
                ),
            )
        )

        new_val_loss, new_val_accuracy = (
            evaluate(
                model=model,
                loader=new_val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                description=(
                    f"Epoch {epoch:02d} NEW val"
                ),
            )
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

        # Scheduler advances only after the current epoch.
        scheduler.step()

        # --------------------------------------------------------------
        # Console summary
        # --------------------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}"
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
            f"  LR layer4:        "
            f"{current_lrs[0]:.8f}"
        )

        print(
            f"  LR classifier:    "
            f"{current_lrs[1]:.8f}"
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

        # Always save complete latest state.
        save_checkpoint(
            checkpoint,
            last_path,
        )

        # Save best checkpoint only when NEW validation
        # accuracy improves.
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
    print("Training finished.")
    print(
        f"Best checkpoint: {best_path}"
    )
    print(
        f"Last checkpoint: {last_path}"
    )
    print()


if __name__ == "__main__":
    main()