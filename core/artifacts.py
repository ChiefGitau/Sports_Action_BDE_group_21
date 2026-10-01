"""Fixed file formats that modules use to talk to each other. Nobody invents another format.

    logits_{split}.pt   {'logits': float32 [E, N, C], 'y': int64 [N]}
    flops.json          {'tool': str, 'flops_per_exit': [int, ...], 'params': int, 'size_mb': float}
    thresholds.json     {budget_level (str): [threshold per exit (float)]}
    results.csv         one row per operating point, columns = RESULT_COLUMNS
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor


# ---------- logits ----------

def save_logits(path: str | Path, logits: Tensor, y: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits must be [E, N, C], got {tuple(logits.shape)}")
    if y.shape != (logits.shape[1],):
        raise ValueError("y must have one label per window (logits.shape[1])")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits.float().cpu(), "y": y.long().cpu()}, path)


def load_logits(path: str | Path) -> tuple[Tensor, Tensor]:
    d = torch.load(path)
    return d["logits"], d["y"]


# ---------- flops.json ----------

def save_flops(path: str | Path, flops: dict) -> None:
    required = {"flops_per_exit", "params", "size_mb"}
    if missing := required - flops.keys():
        raise ValueError(f"flops.json missing keys: {sorted(missing)}")
    _write_json(path, flops)


def load_flops(path: str | Path) -> dict:
    return _read_json(path)


# ---------- thresholds.json ----------

def save_thresholds(path: str | Path, thresholds: dict) -> None:
    """thresholds: {budget_level: [threshold per exit]}. Keys are stored as strings (JSON)."""
    out = {str(k): [float(t) for t in v] for k, v in thresholds.items()}
    _write_json(path, out)


def load_thresholds(path: str | Path) -> dict[str, list[float]]:
    return _read_json(path)


# ---------- results.csv ----------

@dataclass
class ResultRow:
    model: str          # e.g. "cnn_full", "cnn_pruned", "earlyexit"
    variant: str        # e.g. "fp32", "int8", "fp32+kd"
    budget: str         # budget level ("100", "50", "20") or "static" for fixed models
    split: str          # "val" or "test"
    flops_mean: float   # mean FLOPs per window at this operating point
    acc: float
    bal_acc: float
    f1: float
    size_mb: float
    latency_ms: float = float("nan")   # optional, CPU batch size 1


RESULT_COLUMNS = [f.name for f in fields(ResultRow)]


def append_result(path: str | Path, row: ResultRow) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(row))


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv columns {list(df.columns)} != {RESULT_COLUMNS}")
    return df


# ---------- helpers ----------

def _write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
