"""Build models from the config and save/load checkpoints.

Checkpoint format (outputs/{name}/model.pt):
    {'kind': 'static' | 'early_exit', 'arch': {constructor kwargs}, 'variant': str,
     'state_dict': ..., 'module': nn.Module (only for pruned models, whose widths the
     'arch' kwargs can no longer describe)}
"""
from __future__ import annotations

from pathlib import Path

import torch

from core.config import AttrDict

from .baseline import StaticCNN
from .cnn import ExitCNN
from .early_exit import EarlyExitCNN

KINDS = {"static": StaticCNN, "early_exit": EarlyExitCNN}


def arch_from_config(cfg: AttrDict, kind: str, **overrides) -> dict:
    m = cfg.model
    arch = dict(
        max_people=cfg.data.max_people,
        num_classes=cfg.data.num_classes,
        channels=m.channels,
        kernel_size=m.kernel_size,
        num_blocks=m.num_blocks,
        dropout=m.dropout,
        head_pool=m.head_pool,
        features=str(m.get("features", "pos")),
        fps=float(cfg.data.fps),
    )
    if kind == "early_exit":
        arch["exit_after"] = list(m.exit_after)
    arch.update(overrides)
    return arch


def build_model(cfg: AttrDict, kind: str, **overrides) -> ExitCNN:
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}; choose from {sorted(KINDS)}")
    return KINDS[kind](**arch_from_config(cfg, kind, **overrides))


def save_checkpoint(path: str | Path, model: ExitCNN, kind: str, arch: dict, variant: str, pruned: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"kind": kind, "arch": arch, "variant": variant, "state_dict": model.state_dict()}
    if pruned:
        ckpt["module"] = model          # widths differ from arch; ship the whole module
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path) -> tuple[ExitCNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "module" in ckpt:
        model = ckpt["module"]
    else:
        model = KINDS[ckpt["kind"]](**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k not in ("state_dict", "module")}
