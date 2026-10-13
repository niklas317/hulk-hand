#!/usr/bin/env python3

"""Class-balanced NEW-only sampling for DINOv2 training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler

from dataset import GestureDataset


BATCH_SIZE = 32
BATCHES_PER_EPOCH = 281
SAMPLES_PER_EPOCH = BATCH_SIZE * BATCHES_PER_EPOCH


class BalancedNewBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: GestureDataset,
        batch_size: int = BATCH_SIZE,
        batches_per_epoch: int = BATCHES_PER_EPOCH,
        seed: int = 42,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("NEW training dataset is empty.")
        if batch_size % 4 != 0:
            raise ValueError("Batch size must be divisible by four classes.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            grouped[sample.label].append(index)
        self.indices_by_class = dict(grouped)
        self.class_labels = sorted(self.indices_by_class)
        if len(self.class_labels) != 4:
            raise ValueError(f"Expected four NEW classes, found {self.class_labels}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be >= 0.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_class = self.batch_size // len(self.class_labels)
        samples_per_class = self.batches_per_epoch * per_class
        sequences: dict[int, list[int]] = {}

        for label in self.class_labels:
            pool = list(self.indices_by_class[label])
            sequence: list[int] = []
            while len(sequence) < samples_per_class:
                rng.shuffle(pool)
                sequence.extend(pool)
            sequences[label] = sequence[:samples_per_class]

        for batch_index in range(self.batches_per_epoch):
            batch: list[int] = []
            start = batch_index * per_class
            for label in self.class_labels:
                batch.extend(sequences[label][start:start + per_class])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


def build_new_only_dataset_and_sampler(
    new_train: GestureDataset,
    seed: int = 42,
) -> tuple[GestureDataset, BalancedNewBatchSampler]:
    return new_train, BalancedNewBatchSampler(new_train, seed=seed)
