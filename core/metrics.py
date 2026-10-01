"""Shared metrics: accuracy, balanced accuracy, F1. Report all three; the data is imbalanced.

For binary tasks F1 is for the positive class (1 = action). For more classes it is macro F1.
"""
from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(preds: Tensor, y: Tensor, num_classes: int) -> Tensor:
    """[C, C] with rows = true class, cols = predicted class."""
    idx = y.long() * num_classes + preds.long()
    return torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)


def compute_metrics(logits: Tensor, y: Tensor, num_classes: int | None = None) -> dict[str, float]:
    """logits [N, C], y [N] -> {'acc', 'bal_acc', 'f1'}."""
    C = num_classes or logits.shape[-1]
    preds = logits.argmax(dim=-1)
    cm = confusion_matrix(preds, y, C).double()

    tp = cm.diag()
    support = cm.sum(dim=1)              # true count per class
    predicted = cm.sum(dim=0)            # predicted count per class

    acc = (tp.sum() / cm.sum().clamp(min=1)).item()

    present = support > 0
    recall = tp / support.clamp(min=1)
    bal_acc = recall[present].mean().item() if present.any() else 0.0

    precision = tp / predicted.clamp(min=1)
    f1_per_class = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    f1 = f1_per_class[1].item() if C == 2 else f1_per_class[present].mean().item()

    return {"acc": acc, "bal_acc": bal_acc, "f1": f1}
