"""Cost measurement: FLOPs per exit, parameter count, size on disk.

One tool for everyone: torch.utils.flop_counter.FlopCounterMode (built into PyTorch >= 2.1).
It counts FLOPs of matmul/linear/conv/attention ops (2 FLOPs per multiply-add) and ignores
cheap elementwise ops (ReLU, add, softmax). Always report which tool produced the number.
"""
from __future__ import annotations

import io
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from .model import ExitModel

FLOPS_TOOL = "torch.utils.flop_counter.FlopCounterMode"


def count_flops(fn: Callable, *args) -> int:
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        fn(*args)
    return int(counter.get_total_flops())


def flops_per_exit(model: ExitModel, x: Tensor, mask: Tensor) -> list[int]:
    """FLOPs to reach each exit for a *single* window (batch size 1, the streaming case)."""
    was_training = model.training
    model.eval()
    x1, m1 = x[:1], mask[:1]
    out = [count_flops(model.forward_until, x1, m1, e) for e in range(model.num_exits)]
    model.train(was_training)
    return out


def param_count(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized size of the state_dict in MB, i.e. what would ship to the device."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def profile(model: ExitModel, x: Tensor, mask: Tensor) -> dict:
    """Everything M4 needs for flops.json in one call."""
    return {
        "tool": FLOPS_TOOL,
        "flops_per_exit": flops_per_exit(model, x, mask),
        "params": param_count(model),
        "size_mb": state_dict_size_mb(model),
    }
