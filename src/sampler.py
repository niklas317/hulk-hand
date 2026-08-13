#!/usr/bin/env python3

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import ConcatDataset, Sampler

from dataset import GestureDataset


BATCH_SIZE = 128
SAMPLES_PER_SOURCE_PER_BATCH = 64

BATCHES_PER_EPOCH = 141

SAMPLES_PER_EPOCH = (
    BATCH_SIZE * BATCHES_PER_EPOCH
)

SAMPLES_PER_SOURCE_PER_EPOCH = (
    SAMPLES_PER_SOURCE_PER_BATCH
    * BATCHES_PER_EPOCH
)


class BalancedOldNewBatchSampler(
    Sampler[list[int]]
):
    """
    Batch sampler for hulk-hand training.

    Every batch contains exactly:

        64 OLD samples
        64 NEW samples

    Over one epoch, OLD and NEW are independently
    class-balanced as evenly as mathematically possible.

    With the default configuration:

        141 batches / epoch
        128 samples / batch

        = 18,048 samples / epoch

        9,024 OLD
        9,024 NEW

    The sampler is intended to be used with:

        ConcatDataset([
            old_train_dataset,
            new_train_dataset,
        ])

    Global indices are generated accordingly.
    """

    def __init__(
        self,
        old_dataset: GestureDataset,
        new_dataset: GestureDataset,
        seed: int = 42,
    ) -> None:

        if len(old_dataset) == 0:
            raise ValueError(
                "OLD training dataset is empty."
            )

        if len(new_dataset) == 0:
            raise ValueError(
                "NEW training dataset is empty."
            )

        self.old_dataset = old_dataset
        self.new_dataset = new_dataset

        self.seed = seed
        self.epoch = 0

        self.old_indices_by_class = (
            self._group_indices_by_class(
                old_dataset
            )
        )

        self.new_indices_by_class = (
            self._group_indices_by_class(
                new_dataset
            )
        )

        old_classes = set(
            self.old_indices_by_class.keys()
        )

        new_classes = set(
            self.new_indices_by_class.keys()
        )

        if old_classes != new_classes:
            raise ValueError(
                "OLD and NEW datasets do not "
                "contain the same classes.\n"
                f"OLD labels: {sorted(old_classes)}\n"
                f"NEW labels: {sorted(new_classes)}"
            )

        self.class_labels = sorted(
            old_classes
        )

        if not self.class_labels:
            raise ValueError(
                "No classes found."
            )

        self.new_offset = len(
            self.old_dataset
        )

    @staticmethod
    def _group_indices_by_class(
        dataset: GestureDataset,
    ) -> dict[int, list[int]]:
        """
        Group local dataset indices by class label.
        """

        grouped: dict[int, list[int]] = (
            defaultdict(list)
        )

        for index, sample in enumerate(
            dataset.samples
        ):
            grouped[sample.label].append(
                index
            )

        return dict(grouped)

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Set the current epoch.

        train.py should call this once before
        every epoch so that a new deterministic
        random sample order is generated.
        """

        if epoch < 0:
            raise ValueError(
                "Epoch must be >= 0."
            )

        self.epoch = epoch

    def _calculate_class_quotas(
        self,
        total_samples: int,
        rng: random.Random,
    ) -> dict[int, int]:
        """
        Divide total_samples as evenly as possible
        across all classes.

        If the number is not exactly divisible,
        the remaining samples are randomly assigned
        to classes for this epoch.
        """

        num_classes = len(
            self.class_labels
        )

        base = (
            total_samples
            // num_classes
        )

        remainder = (
            total_samples
            % num_classes
        )

        quotas = {
            label: base
            for label in self.class_labels
        }

        if remainder > 0:

            remainder_classes = list(
                self.class_labels
            )

            rng.shuffle(
                remainder_classes
            )

            for label in remainder_classes[
                :remainder
            ]:
                quotas[label] += 1

        return quotas

    @staticmethod
    def _sample_from_class(
        available_indices: list[int],
        count: int,
        rng: random.Random,
    ) -> list[int]:
        """
        Select count indices from one class.

        If enough unique samples exist, sampling is
        performed without replacement.

        If count exceeds the available number of
        samples, shuffled passes through the class
        are repeated until the requested count is
        reached.
        """

        if not available_indices:
            raise ValueError(
                "Cannot sample from an empty class."
            )

        selected: list[int] = []

        while len(selected) < count:

            pool = list(
                available_indices
            )

            rng.shuffle(
                pool
            )

            remaining = (
                count - len(selected)
            )

            selected.extend(
                pool[:remaining]
            )

        return selected

    def _build_source_epoch_indices(
        self,
        indices_by_class: dict[int, list[int]],
        total_samples: int,
        rng: random.Random,
    ) -> list[int]:
        """
        Build one class-balanced index sequence
        for OLD or NEW data.
        """

        quotas = (
            self._calculate_class_quotas(
                total_samples=total_samples,
                rng=rng,
            )
        )

        selected: list[int] = []

        for label in self.class_labels:

            class_indices = (
                indices_by_class[label]
            )

            class_count = (
                quotas[label]
            )

            sampled = (
                self._sample_from_class(
                    available_indices=class_indices,
                    count=class_count,
                    rng=rng,
                )
            )

            selected.extend(
                sampled
            )

        if len(selected) != total_samples:
            raise RuntimeError(
                "Internal sampler error: "
                "incorrect epoch sample count."
            )

        rng.shuffle(
            selected
        )

        return selected

    def __iter__(
        self,
    ) -> Iterator[list[int]]:

        rng = random.Random(
            self.seed + self.epoch
        )

        old_indices = (
            self._build_source_epoch_indices(
                indices_by_class=(
                    self.old_indices_by_class
                ),
                total_samples=(
                    SAMPLES_PER_SOURCE_PER_EPOCH
                ),
                rng=rng,
            )
        )

        new_indices = (
            self._build_source_epoch_indices(
                indices_by_class=(
                    self.new_indices_by_class
                ),
                total_samples=(
                    SAMPLES_PER_SOURCE_PER_EPOCH
                ),
                rng=rng,
            )
        )

        for batch_index in range(
            BATCHES_PER_EPOCH
        ):

            start = (
                batch_index
                * SAMPLES_PER_SOURCE_PER_BATCH
            )

            end = (
                start
                + SAMPLES_PER_SOURCE_PER_BATCH
            )

            old_batch = (
                old_indices[start:end]
            )

            # NEW indices need an offset because
            # DataLoader operates on ConcatDataset:
            #
            # [ OLD ][ NEW ]
            #
            new_batch = [
                index + self.new_offset
                for index
                in new_indices[start:end]
            ]

            batch = (
                old_batch
                + new_batch
            )

            # Avoid a fixed OLD-first / NEW-second
            # ordering inside every batch.
            rng.shuffle(
                batch
            )

            if len(batch) != BATCH_SIZE:
                raise RuntimeError(
                    "Internal sampler error: "
                    "incorrect batch size."
                )

            yield batch

    def __len__(
        self,
    ) -> int:

        return BATCHES_PER_EPOCH


def build_training_dataset_and_sampler(
    old_train: GestureDataset,
    new_train: GestureDataset,
    seed: int = 42,
) -> tuple[
    ConcatDataset,
    BalancedOldNewBatchSampler,
]:
    """
    Convenience function used by train.py.

    Returns:

        combined training dataset
        balanced OLD/NEW batch sampler
    """

    combined_dataset = ConcatDataset(
        [
            old_train,
            new_train,
        ]
    )

    sampler = (
        BalancedOldNewBatchSampler(
            old_dataset=old_train,
            new_dataset=new_train,
            seed=seed,
        )
    )

    return (
        combined_dataset,
        sampler,
    )