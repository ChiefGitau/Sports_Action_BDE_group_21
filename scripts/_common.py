"""Shared argparse bits for the M2/M3 training scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def base_parser(description: str, default_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--name", default=default_name, help="output folder under outputs/ and model column in results.csv")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument("--seed", type=int, default=None, help="override seed")
    p.add_argument("--no-export", action="store_true", help="train only; do not write outputs/ or results.csv")
    return p


def prepare(args):
    import core

    cfg = core.load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    core.set_seed(cfg.seed)
    return cfg
