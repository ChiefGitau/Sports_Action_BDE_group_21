"""M3 early-exit model: shared trunk with one lightweight head after each block in `exit_after`.

Trained jointly (weighted per-exit cross-entropy) with optional knowledge distillation where
every early exit also matches the final head's softened outputs; see models/train.py.
"""
from __future__ import annotations

from .cnn import ExitCNN


class EarlyExitCNN(ExitCNN):
    def __init__(
        self,
        max_people: int,
        num_classes: int = 2,
        channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.1,
        exit_after: tuple[int, ...] | list[int] = (1, 2, 3, 4),
        head_pool: str = "mean",
    ):
        if len(exit_after) < 2:
            raise ValueError("an early-exit model needs at least two exits; use StaticCNN otherwise")
        super().__init__(
            max_people=max_people,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout=dropout,
            exit_after=exit_after,
            head_pool=head_pool,
        )
