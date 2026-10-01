"""Config loading. One YAML file (configs/default.yaml) drives every module."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class AttrDict(dict):
    """dict with attribute access, applied recursively (cfg.data.fps)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def from_dict(cls, d: dict) -> "AttrDict":
        out = cls()
        for k, v in d.items():
            out[k] = cls.from_dict(v) if isinstance(v, dict) else v
        return out


def load_config(path: str | Path | None = None) -> AttrDict:
    """Load the YAML config and return it as a nested AttrDict."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AttrDict.from_dict(raw)


def resolve_path(cfg: AttrDict, key: str) -> Path:
    """Return cfg.paths[key] as an absolute path anchored at the repo root."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else REPO_ROOT / p


def window_frames(cfg: AttrDict) -> int:
    """T in the data contract: frames per window, derived from fps and window length."""
    return max(1, round(cfg.data.fps * cfg.data.window_seconds))


def stride_frames(cfg: AttrDict) -> int:
    return max(1, round(cfg.data.fps * cfg.data.stride_seconds))
