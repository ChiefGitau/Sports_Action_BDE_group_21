"""M2 static baselines: the full trunk with a single exit at the end.

    cnn_full    StaticCNN trained from scratch
    cnn_pruned  cnn_full physically pruned to `prune.ratio` of its channels (models/prune.py),
                then fine-tuned. Physical removal, not masking, so counted FLOPs really drop.

Once trained and exported these are frozen: nobody changes baselines afterwards.
"""
from __future__ import annotations

from .cnn import ExitCNN


class StaticCNN(ExitCNN):
    """Same trunk as the early-exit model, one head after the last block."""

    def __init__(
        self,
        max_people: int,
        num_classes: int = 2,
        channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.1,
        head_pool: str = "mean",
    ):
        super().__init__(
            max_people=max_people,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout=dropout,
            exit_after=[num_blocks],
            head_pool=head_pool,
        )
