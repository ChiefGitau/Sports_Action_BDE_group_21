"""The data contract shared by every module.

A split is a dict with exactly these keys (saved with torch.save as data/processed/{split}.pt):

    X     float32 [N, T, P, 2]   windows x frames x people x (x, y) position
    mask  bool    [N, T, P]      True where a person slot is actually present
    y     int64   [N]            0 = no action, 1 = action
    meta  dict                   match_id: list[str] (len N)
                                 half: int64 [N]
                                 start_frame: int64 [N]

Absent people must have mask=False and X=0. Positions should be normalised to the pitch
(M1 decides the exact normalisation; document it in the data report).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "val", "test")
META_KEYS = ("match_id", "half", "start_frame")


def validate_split(d: dict, num_classes: int | None = None) -> None:
    """Raise a ValueError if `d` violates the data contract."""
    for key in ("X", "mask", "y", "meta"):
        if key not in d:
            raise ValueError(f"missing key '{key}'")
    X, mask, y, meta = d["X"], d["mask"], d["y"], d["meta"]

    if X.dtype != torch.float32 or X.ndim != 4 or X.shape[-1] != 2:
        raise ValueError(f"X must be float32 [N, T, P, 2], got {X.dtype} {tuple(X.shape)}")
    N, T, P, _ = X.shape
    if mask.dtype != torch.bool or tuple(mask.shape) != (N, T, P):
        raise ValueError(f"mask must be bool [N, T, P]={(N, T, P)}, got {mask.dtype} {tuple(mask.shape)}")
    if y.dtype != torch.int64 or tuple(y.shape) != (N,):
        raise ValueError(f"y must be int64 [N]={(N,)}, got {y.dtype} {tuple(y.shape)}")
    if num_classes is not None and N > 0 and (y.min() < 0 or y.max() >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes}), got range [{y.min()}, {y.max()}]")

    for k in META_KEYS:
        if k not in meta:
            raise ValueError(f"meta missing '{k}'")
    if len(meta["match_id"]) != N:
        raise ValueError("meta.match_id must have one entry per window")
    for k in ("half", "start_frame"):
        if meta[k].dtype != torch.int64 or tuple(meta[k].shape) != (N,):
            raise ValueError(f"meta.{k} must be int64 [N]")

    if not torch.isfinite(X).all():
        raise ValueError("X contains NaN/inf")
    if (X[~mask] != 0).any():
        raise ValueError("X must be zero where mask is False")


def save_split(d: dict, path: str | Path, num_classes: int | None = None) -> None:
    validate_split(d, num_classes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)


def load_split(path: str | Path, num_classes: int | None = None) -> dict:
    d = torch.load(path)
    validate_split(d, num_classes)
    return d


def make_loader(d: dict, batch_size: int, shuffle: bool = False) -> DataLoader:
    """DataLoader yielding (X, mask, y) batches. Meta is not batched; use it from the dict."""
    return DataLoader(TensorDataset(d["X"], d["mask"], d["y"]), batch_size=batch_size, shuffle=shuffle)


def class_balance(y: Tensor, num_classes: int) -> list[float]:
    """Fraction of windows per class."""
    counts = torch.bincount(y, minlength=num_classes).float()
    return (counts / counts.sum().clamp(min=1)).tolist()
