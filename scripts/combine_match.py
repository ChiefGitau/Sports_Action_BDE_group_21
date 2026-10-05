from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# Make repository root importable when run as:
# python scripts\combine_match.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config
from core.data import load_split, save_split


def combine_splits(first: dict, second: dict) -> dict:
    """
    Combine two datasets that follow the Core data contract.

    X:
        [N, T, P, 2]

    mask:
        [N, T, P]

    y:
        [N]

    meta:
        match_id: list[str]
        half: int64 [N]
        start_frame: int64 [N]
    """

    # Check that tensor dimensions other than N agree.
    if first["X"].shape[1:] != second["X"].shape[1:]:
        raise ValueError(
            "X shapes are incompatible: "
            f"{tuple(first['X'].shape)} vs "
            f"{tuple(second['X'].shape)}"
        )

    if first["mask"].shape[1:] != second["mask"].shape[1:]:
        raise ValueError(
            "mask shapes are incompatible: "
            f"{tuple(first['mask'].shape)} vs "
            f"{tuple(second['mask'].shape)}"
        )

    combined = {
        "X": torch.cat(
            [first["X"], second["X"]],
            dim=0,
        ),
        "mask": torch.cat(
            [first["mask"], second["mask"]],
            dim=0,
        ),
        "y": torch.cat(
            [first["y"], second["y"]],
            dim=0,
        ),
        "meta": {
            "match_id": (
                list(first["meta"]["match_id"])
                + list(second["meta"]["match_id"])
            ),
            "half": torch.cat(
                [
                    first["meta"]["half"],
                    second["meta"]["half"],
                ],
                dim=0,
            ),
            "start_frame": torch.cat(
                [
                    first["meta"]["start_frame"],
                    second["meta"]["start_frame"],
                ],
                dim=0,
            ),
        },
    }

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine two Core-format match halves."
    )

    parser.add_argument(
        "--first",
        type=Path,
        required=True,
        help="First Core-format .pt file",
    )

    parser.add_argument(
        "--second",
        type=Path,
        required=True,
        help="Second Core-format .pt file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output combined .pt file",
    )

    args = parser.parse_args()

    cfg = load_config()

    print("Loading first half:")
    print(args.first)

    first = load_split(
        args.first,
        num_classes=cfg.data.num_classes,
    )

    print("Loading second half:")
    print(args.second)

    second = load_split(
        args.second,
        num_classes=cfg.data.num_classes,
    )

    print("\n=== INPUT SHAPES ===")
    print("first X:", tuple(first["X"].shape))
    print("second X:", tuple(second["X"].shape))

    combined = combine_splits(first, second)

    print("\n=== COMBINED SHAPES ===")
    print(
        "X:",
        tuple(combined["X"].shape),
        combined["X"].dtype,
    )
    print(
        "mask:",
        tuple(combined["mask"].shape),
        combined["mask"].dtype,
    )
    print(
        "y:",
        tuple(combined["y"].shape),
        combined["y"].dtype,
    )

    print("\n=== LABEL BALANCE ===")
    y = combined["y"]

    print("y=0:", int((y == 0).sum()))
    print("y=1:", int((y == 1).sum()))
    print(
        "positive fraction:",
        round(float((y == 1).float().mean()), 4),
    )

    print("\n=== HALF COUNTS ===")
    halves = combined["meta"]["half"]

    print("half 1:", int((halves == 1).sum()))
    print("half 2:", int((halves == 2).sum()))

    match_ids = combined["meta"]["match_id"]

    print("\n=== MATCH IDS ===")
    print("unique match ids:", sorted(set(match_ids)))

    print("\nValidating and saving...")

    save_split(
        combined,
        args.output,
        num_classes=cfg.data.num_classes,
    )

    print("saved:", args.output)
    print("\nCombined match dataset completed successfully.")


if __name__ == "__main__":
    main()