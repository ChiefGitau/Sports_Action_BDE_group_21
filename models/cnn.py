"""Shared temporal 1D-CNN used by the M2 baselines and the M3 early-exit model.

Per frame, the P people are flattened to P * 3 features (x, y, present) so absent slots are
explicit. A 1x1 conv embeds them into `channels`, then `num_blocks` conv blocks run over the
T frames. An exit head is `pool over frames -> Linear(channels, C)`: deliberately light so
exits do not cancel the savings they are meant to give.

    ExitCNN(exit_after=[num_blocks])      -> static baseline (M2)
    ExitCNN(exit_after=[1, 2, 3, 4])      -> early-exit model (M3)

Both share exactly the same trunk code, so the comparison in M7 is between training/exit
strategies, not architectures.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from core.model import ExitModel


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        return self.drop(self.act(self.bn(self.conv(h))))


class PoolHead(nn.Module):
    """Exit head: pool over frames + one Linear. [N, C, T] -> [N, num_classes]."""

    def __init__(self, channels: int, num_classes: int, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"head_pool must be 'mean' or 'max', got {pool!r}")
        self.pool = pool
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, h: Tensor) -> Tensor:
        p = h.mean(dim=2) if self.pool == "mean" else h.amax(dim=2)
        return self.fc(p)


FEATURES_PER_PERSON = {"pos": 3, "pos+vel": 5, "vel": 3}


def featurize(x: Tensor, mask: Tensor, features: str = "pos", fps: float = 25.0) -> Tensor:
    """[N, T, P, 2] + [N, T, P] -> [N, F*P, T], always with a presence bit per person.

        pos      (x, y, present)                    F = 3
        pos+vel  (x, y, vx, vy, present)            F = 5
        vel      (vx, vy, present)                  F = 3   positions dropped entirely

    Velocities are per-frame displacements in pitch units per second. They are what players
    are *doing*; absolute positions depend on pitch orientation, camera and which end a team
    defends, and change meaning between halves and matches."""
    if features not in FEATURES_PER_PERSON:
        raise ValueError(f"features must be one of {sorted(FEATURES_PER_PERSON)}, got {features!r}")
    m = mask.unsqueeze(-1).to(x.dtype)
    x = x * m
    parts = []
    if features in ("pos", "pos+vel"):
        parts.append(x)
    if features in ("pos+vel", "vel"):
        v = torch.zeros_like(x)
        v[:, 1:] = (x[:, 1:] - x[:, :-1]) * fps * (m[:, 1:] * m[:, :-1])
        parts.append(v)
    parts.append(m)
    feats = torch.cat(parts, dim=-1)                                   # [N, T, P, F]
    return feats.flatten(2).transpose(1, 2).contiguous()


class ExitCNN(ExitModel):
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
        features: str = "pos",
        fps: float = 25.0,
    ):
        super().__init__()
        self.features, self.fps = str(features), float(fps)
        exit_after = sorted(int(e) for e in exit_after)
        if not exit_after or len(set(exit_after)) != len(exit_after):
            raise ValueError("exit_after must be a non-empty list of distinct block indices")
        if exit_after[0] < 1 or exit_after[-1] != num_blocks:
            raise ValueError(f"exit_after must lie in [1, num_blocks] and end at num_blocks={num_blocks}")
        self.exit_after = exit_after
        self.embed = nn.Conv1d(max_people * FEATURES_PER_PERSON[self.features], channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ConvBlock(channels, channels, kernel_size, dropout) for _ in range(num_blocks)]
        )
        self.heads = nn.ModuleList([PoolHead(channels, num_classes, head_pool) for _ in exit_after])

    # ---- ExitModel interface ----

    @property
    def num_exits(self) -> int:
        return len(self.exit_after)

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        h = self.embed(featurize(x, mask, self.features, self.fps))
        outs: list[Tensor] = []
        head = 0
        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            if head < len(self.exit_after) and i == self.exit_after[head]:
                outs.append(self.heads[head](h))
                head += 1
        return outs

    def forward_until(self, x: Tensor, mask: Tensor, exit_idx: int) -> Tensor:
        """Exactly the computation a deployed device runs to answer at exit `exit_idx`.

        That includes every earlier head and its confidence check: the device evaluated
        them, found them not confident enough, and continued. Leaving them out would
        under-report the controller overhead (adaptivity lecture, slide 34). Note that
        torch's FLOP counter does not count the softmax itself, only the Linear heads.
        """
        self._check_exit(exit_idx)
        h = self.embed(featurize(x, mask, self.features, self.fps))
        head = 0
        for i, block in enumerate(self.blocks[: self.exit_after[exit_idx]], start=1):
            h = block(h)
            if i == self.exit_after[head]:
                if head == exit_idx:
                    return self.heads[head](h)
                self.heads[head](h).softmax(dim=-1)     # rejected exit: computed, then discarded
                head += 1
        raise RuntimeError("unreachable: exit_after must contain exit_idx")   # pragma: no cover

    # ---- introspection used by M4 (weight sharing) and pruning ----

    def trunk_modules(self) -> list[nn.Module]:
        return [self.embed, *self.blocks]

    def channel_widths(self) -> list[int]:
        """Output channels of embed and every block (changes after physical pruning)."""
        return [self.embed.out_channels] + [b.conv.out_channels for b in self.blocks]
