"""M2 pruned baseline: structured, *physical* channel pruning with Torch-Pruning (DepGraph).

Masking would leave the FLOP count untouched; DepGraph rewrites the Conv1d/BatchNorm1d/Linear
shapes so the pruned model is genuinely smaller. Exit heads are kept as roots-to-ignore so
their output size (num_classes) never changes; their input width follows the trunk.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .cnn import ExitCNN


def prune_channels(model: ExitCNN, x: Tensor, mask: Tensor, ratio: float) -> ExitCNN:
    """Remove `ratio` of the channels of every trunk conv (L2-magnitude importance), in place."""
    try:
        import torch_pruning as tp
    except ImportError as e:   # pragma: no cover
        raise ImportError("pip install torch-pruning  (see requirements.txt)") from e
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"prune ratio must be in (0, 1), got {ratio}")

    model = model.cpu().eval()
    before = model.channel_widths()
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=(x[:2].cpu(), mask[:2].cpu()),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[h.fc for h in model.heads],
        root_module_types=[nn.Conv1d],
    )
    pruner.step()
    after = model.channel_widths()
    if any(a >= b for a, b in zip(after, before)):
        raise RuntimeError(f"pruning did not shrink the trunk: {before} -> {after}")
    model.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def trunk_width_report(model: ExitCNN) -> str:
    return " -> ".join(str(w) for w in model.channel_widths())
