"""Mock data in exactly the Core data contract, so modules can run before M1 delivers.

The data is random but carries a weak signal (action windows move more), so training loops
can be checked for "does it learn anything at all" without pretending to be realistic.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import AttrDict, resolve_path, window_frames
from .data import SPLITS, save_split


def make_mock_split(
    n: int,
    T: int,
    P: int,
    match_ids: list[str],
    positive_fraction: float = 0.17,
    stride: int = 12,
    seed: int = 0,
) -> dict:
    g = torch.Generator().manual_seed(seed)

    y = (torch.rand(n, generator=g) < positive_fraction).long()

    # Random walks per person, starting from a uniform position on a unit pitch.
    start = torch.rand(n, 1, P, 2, generator=g)
    step_scale = torch.where(y.bool(), torch.tensor(0.03), torch.tensor(0.01)).view(n, 1, 1, 1)
    steps = torch.randn(n, T, P, 2, generator=g) * step_scale
    X = (start + steps.cumsum(dim=1)).clamp(0.0, 1.0)

    # Each window has a random number of present people (between 60% and all), same across frames.
    n_present = torch.randint(int(0.6 * P), P + 1, (n,), generator=g)
    mask = torch.arange(P).view(1, 1, P) < n_present.view(n, 1, 1)
    mask = mask.expand(n, T, P).clone()
    X = X * mask.unsqueeze(-1)

    idx = torch.randint(len(match_ids), (n,), generator=g)
    meta = {
        "match_id": [match_ids[i] for i in idx.tolist()],
        "half": torch.randint(1, 3, (n,), generator=g),
        "start_frame": torch.arange(n, dtype=torch.int64) * stride,
    }
    return {"X": X.float(), "mask": mask, "y": y, "meta": meta}


def write_mock_dataset(cfg: AttrDict, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write mock {train,val,test}.pt following the config. Returns the paths written."""
    out_dir = Path(out_dir) if out_dir is not None else resolve_path(cfg, "processed")
    T = window_frames(cfg)
    P = cfg.data.max_people
    paths = {}
    for i, split in enumerate(SPLITS):
        d = make_mock_split(
            n=cfg.mock[f"n_{split}"],
            T=T,
            P=P,
            match_ids=list(cfg.data.splits[split]),
            positive_fraction=cfg.mock.positive_fraction,
            seed=cfg.seed + i,
        )
        paths[split] = out_dir / f"{split}.pt"
        save_split(d, paths[split], num_classes=cfg.data.num_classes)
    return paths
