"""The model interface every model in this project implements.

    forward(x, mask)                 -> list[Tensor[N, C]]   one entry per exit, shallow first
    num_exits                        -> int                  static baselines return 1
    forward_until(x, mask, exit_idx) -> Tensor[N, C]         run only what exit `exit_idx` needs

`forward_until` must execute exactly the computation a deployed device would run to produce
that exit, because M4 measures per-exit FLOPs by calling it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ExitModel(nn.Module, ABC):
    @property
    @abstractmethod
    def num_exits(self) -> int: ...

    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]: ...

    @abstractmethod
    def forward_until(self, x: Tensor, mask: Tensor, exit_idx: int) -> Tensor: ...

    def _check_exit(self, exit_idx: int) -> None:
        if not 0 <= exit_idx < self.num_exits:
            raise IndexError(f"exit_idx {exit_idx} out of range for {self.num_exits} exits")


def stack_exit_logits(outs: list[Tensor]) -> Tensor:
    """list of [N, C] (one per exit) -> [E, N, C], the layout of logits_{split}.pt."""
    return torch.stack(outs, dim=0)


class DummyExitModel(ExitModel):
    """Minimal MLP trunk with one lightweight head per block. Exists to exercise the interface
    and to give M4/M5/M7 something runnable; M3 replaces it with the real early-exit model.

    Trunk:  flatten people -> Linear -> [ReLU -> Linear] x num_exits
    Head i: mean over frames -> Linear(hidden, C)
    """

    def __init__(self, max_people: int, num_classes: int = 2, num_exits: int = 4, hidden: int = 64):
        super().__init__()
        self._num_exits = num_exits
        self.embed = nn.Linear(max_people * 2, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(num_exits)]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(num_exits)])

    @property
    def num_exits(self) -> int:
        return self._num_exits

    def _embed(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x * mask.unsqueeze(-1)              # zero absent people (already true by contract)
        return self.embed(x.flatten(2))         # [N, T, hidden]

    def _head(self, h: Tensor, i: int) -> Tensor:
        return self.heads[i](h.mean(dim=1))     # [N, C]

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        h = self._embed(x, mask)
        outs = []
        for i, block in enumerate(self.blocks):
            h = block(h)
            outs.append(self._head(h, i))
        return outs

    def forward_until(self, x: Tensor, mask: Tensor, exit_idx: int) -> Tensor:
        self._check_exit(exit_idx)
        h = self._embed(x, mask)
        for block in self.blocks[: exit_idx + 1]:
            h = block(h)
        return self._head(h, exit_idx)
