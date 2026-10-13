"""Build data/processed/{train,val,test}.pt from whatever matches have been processed so far.

build_splits.py needs all ten official matches. While they are still downloading, this script
makes the best real-data split it can from the processed matches present:

  * every processed match takes its official role (train / val / test);
  * a role with no match borrows, by *time*, from another: with only train matches, the
    last train match gives its 2nd half to val and test (first/second half of it in time);
    with train + val but no test, the val match's 2nd half becomes test.

Windows are never shuffled across the boundary, so val/test are contiguous stretches of play
the model has not seen. Still a weaker protocol than held-out matches; the report says which.

    python scripts/build_splits_available.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import save_split, validate_split  # noqa: E402

# official SoccerTrack v2 split (same mapping as scripts/build_splits.py)
ROLE = {
    "117092": "train", "118575": "train", "118576": "train", "118577": "train", "118578": "train", "128058": "train",
    "117093": "val", "132877": "val",
    "128057": "test", "132831": "test",
}


def load(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    validate_split(d, num_classes=2)
    return d


def subset(d: dict, idx: torch.Tensor) -> dict:
    return {
        "X": d["X"][idx], "mask": d["mask"][idx], "y": d["y"][idx],
        "meta": {
            "match_id": [d["meta"]["match_id"][i] for i in idx.tolist()],
            "half": d["meta"]["half"][idx],
            "start_frame": d["meta"]["start_frame"][idx],
        },
    }


def concat(parts: list[dict]) -> dict:
    return {
        "X": torch.cat([p["X"] for p in parts]),
        "mask": torch.cat([p["mask"] for p in parts]),
        "y": torch.cat([p["y"] for p in parts]),
        "meta": {
            "match_id": sum((p["meta"]["match_id"] for p in parts), []),
            "half": torch.cat([p["meta"]["half"] for p in parts]),
            "start_frame": torch.cat([p["meta"]["start_frame"] for p in parts]),
        },
    }


def by_half(d: dict, half: int) -> dict:
    idx = torch.nonzero(d["meta"]["half"] == half).flatten()
    idx = idx[torch.argsort(d["meta"]["start_frame"][idx])]
    return subset(d, idx)


def split_in_time(d: dict) -> tuple[dict, dict]:
    """First and second contiguous halves (by window order) of an already time-sorted part."""
    n = d["y"].shape[0]
    return subset(d, torch.arange(0, n // 2)), subset(d, torch.arange(n // 2, n))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = p.parse_args()

    present = {mid: args.processed_dir / f"{mid}.pt" for mid in ROLE if (args.processed_dir / f"{mid}.pt").exists()}
    if not present:
        raise SystemExit(f"no processed matches found in {args.processed_dir}")
    roles: dict[str, list[tuple[str, dict]]] = {"train": [], "val": [], "test": []}
    for mid, path in sorted(present.items()):
        roles[ROLE[mid]].append((mid, load(path)))
    print("processed matches:", {r: [m for m, _ in v] for r, v in roles.items()})

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    source: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    if not roles["train"]:
        raise SystemExit("need at least one processed *train* match (e.g. 117092)")

    if not roles["val"] and not roles["test"]:
        # Single-role case: last train match donates its 2nd half, split in time -> val, test.
        donor_id, donor = roles["train"].pop()
        h1, h2 = by_half(donor, 1), by_half(donor, 2)
        v, t = split_in_time(h2)
        out["train"].append(h1); source["train"].append(f"{donor_id} half 1")
        out["val"].append(v);    source["val"].append(f"{donor_id} half 2, first part (time)")
        out["test"].append(t);   source["test"].append(f"{donor_id} half 2, second part (time)")
    elif not roles["test"]:
        # train + val present: the last val match donates its 2nd half as test.
        donor_id, donor = roles["val"].pop()
        out["val"].append(by_half(donor, 1));  source["val"].append(f"{donor_id} half 1")
        out["test"].append(by_half(donor, 2)); source["test"].append(f"{donor_id} half 2")
    elif not roles["val"]:
        donor_id, donor = roles["test"].pop()
        out["val"].append(by_half(donor, 1));  source["val"].append(f"{donor_id} half 1")
        out["test"].append(by_half(donor, 2)); source["test"].append(f"{donor_id} half 2")

    for role in ("train", "val", "test"):
        for mid, d in roles[role]:
            out[role].append(d); source[role].append(f"{mid} (full match)")

    print()
    for role in ("train", "val", "test"):
        d = concat(out[role])
        save_split(d, args.processed_dir / f"{role}.pt", num_classes=2)
        pos = d["y"].float().mean().item()
        print(f"{role:5s} {d['y'].shape[0]:6d} windows  action {pos:.3f}  <- {'; '.join(source[role])}")
    print(f"\nwrote {args.processed_dir}/{{train,val,test}}.pt")
    if len(present) < len(ROLE):
        print("NOTE: partial data. Re-run after more matches are processed; run build_splits.py once all ten exist.")


if __name__ == "__main__":
    main()
