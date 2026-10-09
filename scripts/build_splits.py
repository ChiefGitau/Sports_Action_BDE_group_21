from __future__ import annotations

import argparse
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Project split defined in adaptive_inference_project_plan.md
#
# M1  = 117092  train
# M2  = 117093  validation
# M3  = 118575  train
# M4  = 118576  train
# M5  = 118577  train
# M6  = 118578  train
# M7  = 128057  test
# M8  = 128058  train
# M9  = 132831  test
# M10 = 132877  validation
# ---------------------------------------------------------------------------

SPLITS = {
    "train": [
        "117092",
        "118575",
        "118576",
        "118577",
        "118578",
        "128058",
    ],
    "val": [
        "117093",
        "132877",
    ],
    "test": [
        "128057",
        "132831",
    ],
}


def load_match(path: Path) -> dict:
    """Load and validate one processed SoccerTrack match."""

    if not path.exists():
        raise FileNotFoundError(f"Processed match does not exist: {path}")

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {"X", "mask", "y", "meta"}
    missing = required - set(data.keys())

    if missing:
        raise ValueError(
            f"{path} is missing required keys: {sorted(missing)}"
        )

    X = data["X"]
    mask = data["mask"]
    y = data["y"]
    meta = data["meta"]

    if not isinstance(X, torch.Tensor):
        raise TypeError(f"{path}: X must be a tensor")

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{path}: mask must be a tensor")

    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{path}: y must be a tensor")

    if not isinstance(meta, dict):
        raise TypeError(f"{path}: meta must be a dictionary")

    n = X.shape[0]

    if mask.shape[0] != n:
        raise ValueError(
            f"{path}: X has {n} samples but mask has "
            f"{mask.shape[0]}"
        )

    if y.shape[0] != n:
        raise ValueError(
            f"{path}: X has {n} samples but y has "
            f"{y.shape[0]}"
        )

    if X.ndim != 4:
        raise ValueError(
            f"{path}: expected X shape [N,T,P,2], got {tuple(X.shape)}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"{path}: expected mask shape [N,T,P], "
            f"got {tuple(mask.shape)}"
        )

    if X.shape[:3] != mask.shape:
        raise ValueError(
            f"{path}: X and mask dimensions do not agree: "
            f"{tuple(X.shape)} vs {tuple(mask.shape)}"
        )

    if X.shape[-1] != 2:
        raise ValueError(
            f"{path}: final X dimension must be 2 for x/y coordinates"
        )

    for key in ("match_id", "half", "start_frame"):
        if key not in meta:
            raise ValueError(
                f"{path}: metadata is missing '{key}'"
            )

    if len(meta["match_id"]) != n:
        raise ValueError(
            f"{path}: match_id metadata length does not match X"
        )

    if len(meta["half"]) != n:
        raise ValueError(
            f"{path}: half metadata length does not match X"
        )

    if len(meta["start_frame"]) != n:
        raise ValueError(
            f"{path}: start_frame metadata length does not match X"
        )

    return data


def validate_match_id(
    data: dict,
    expected_match_id: str,
    path: Path,
) -> None:
    """Verify that the file really belongs to the expected match."""

    ids = {str(x) for x in data["meta"]["match_id"]}

    if ids != {expected_match_id}:
        raise ValueError(
            f"{path}: expected match_id {expected_match_id}, "
            f"but found {sorted(ids)}"
        )


def combine_matches(
    processed_dir: Path,
    match_ids: list[str],
    split_name: str,
) -> dict:
    """Combine complete processed matches into one dataset split."""

    datasets = []

    expected_t = None
    expected_p = None

    print()
    print("=" * 70)
    print(f"BUILDING {split_name.upper()} SPLIT")
    print("=" * 70)

    for match_id in match_ids:
        path = processed_dir / f"{match_id}.pt"

        print(f"\nLoading match {match_id}")
        print("path:", path)

        data = load_match(path)
        validate_match_id(data, match_id, path)

        X = data["X"]

        if expected_t is None:
            expected_t = X.shape[1]
            expected_p = X.shape[2]
        else:
            if X.shape[1] != expected_t:
                raise ValueError(
                    f"{path}: sequence length mismatch. "
                    f"Expected {expected_t}, got {X.shape[1]}"
                )

            if X.shape[2] != expected_p:
                raise ValueError(
                    f"{path}: people dimension mismatch. "
                    f"Expected {expected_p}, got {X.shape[2]}"
                )

        positives = int((data["y"] == 1).sum().item())
        negatives = int((data["y"] == 0).sum().item())

        print("samples:", len(data["y"]))
        print("positive:", positives)
        print("negative:", negatives)

        datasets.append(data)

    print(f"\nCombining {len(datasets)} matches...")

    X = torch.cat(
        [d["X"] for d in datasets],
        dim=0,
    )

    mask = torch.cat(
        [d["mask"] for d in datasets],
        dim=0,
    )

    y = torch.cat(
        [d["y"] for d in datasets],
        dim=0,
    )

    match_id = []

    for d in datasets:
        match_id.extend(
            str(x) for x in d["meta"]["match_id"]
        )

    half = torch.cat(
        [
            torch.as_tensor(d["meta"]["half"])
            for d in datasets
        ],
        dim=0,
    )

    start_frame = torch.cat(
        [
            torch.as_tensor(d["meta"]["start_frame"])
            for d in datasets
        ],
        dim=0,
    )

    combined = {
        "X": X,
        "mask": mask,
        "y": y,
        "meta": {
            "match_id": match_id,
            "half": half,
            "start_frame": start_frame,
        },
    }

    validate_combined(
        combined,
        split_name,
        match_ids,
    )

    return combined


def validate_combined(
    data: dict,
    split_name: str,
    expected_matches: list[str],
) -> None:
    """Run consistency checks on a completed split."""

    X = data["X"]
    mask = data["mask"]
    y = data["y"]
    meta = data["meta"]

    n = X.shape[0]

    if mask.shape[0] != n:
        raise ValueError(
            f"{split_name}: mask sample count mismatch"
        )

    if y.shape[0] != n:
        raise ValueError(
            f"{split_name}: label sample count mismatch"
        )

    if len(meta["match_id"]) != n:
        raise ValueError(
            f"{split_name}: match_id count mismatch"
        )

    if len(meta["half"]) != n:
        raise ValueError(
            f"{split_name}: half count mismatch"
        )

    if len(meta["start_frame"]) != n:
        raise ValueError(
            f"{split_name}: start_frame count mismatch"
        )

    actual_matches = sorted(
        {str(x) for x in meta["match_id"]}
    )

    expected = sorted(expected_matches)

    if actual_matches != expected:
        raise ValueError(
            f"{split_name}: expected matches {expected}, "
            f"but found {actual_matches}"
        )

    unique_labels = set(y.unique().tolist())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{split_name}: labels must be binary, "
            f"found {sorted(unique_labels)}"
        )

    if X.shape[-1] != 2:
        raise ValueError(
            f"{split_name}: X must contain x/y coordinates"
        )

    print()
    print("--- VALIDATION ---")
    print("split:", split_name)
    print("matches:", actual_matches)
    print("samples:", n)
    print("X:", tuple(X.shape), X.dtype)
    print("mask:", tuple(mask.shape), mask.dtype)
    print("y:", tuple(y.shape), y.dtype)


def print_summary(
    split_name: str,
    data: dict,
) -> None:
    """Print useful split statistics."""

    y = data["y"]
    meta = data["meta"]

    positives = int((y == 1).sum().item())
    negatives = int((y == 0).sum().item())

    total = len(y)

    print()
    print("=" * 70)
    print(f"{split_name.upper()} SUMMARY")
    print("=" * 70)

    print("X:", tuple(data["X"].shape), data["X"].dtype)
    print(
        "mask:",
        tuple(data["mask"].shape),
        data["mask"].dtype,
    )
    print("y:", tuple(y.shape), y.dtype)

    print()
    print("samples:", total)
    print("negative:", negatives)
    print("positive:", positives)

    if total:
        print(
            "positive fraction:",
            round(positives / total, 4),
        )

    print(
        "matches:",
        sorted({str(x) for x in meta["match_id"]}),
    )

    half1 = int(
        (torch.as_tensor(meta["half"]) == 1)
        .sum()
        .item()
    )

    half2 = int(
        (torch.as_tensor(meta["half"]) == 2)
        .sum()
        .item()
    )

    print("half 1 windows:", half1)
    print("half 2 windows:", half2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine processed SoccerTrack matches into "
            "train, validation, and test splits."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing the processed match .pt files",
    )

    args = parser.parse_args()

    processed_dir = args.processed_dir

    print("=" * 70)
    print("SOCCERTRACK FINAL DATASET SPLIT BUILDER")
    print("=" * 70)
    print("processed directory:", processed_dir.resolve())

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory does not exist: {processed_dir}"
        )

    # ------------------------------------------------------------------
    # Make sure all ten source matches exist before doing anything.
    # ------------------------------------------------------------------

    all_match_ids = sorted(
        {
            match_id
            for match_ids in SPLITS.values()
            for match_id in match_ids
        }
    )

    print("\nChecking source match files...")

    missing = []

    for match_id in all_match_ids:
        path = processed_dir / f"{match_id}.pt"

        if path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(
                f"  OK  {match_id}.pt "
                f"({size_mb:.2f} MB)"
            )
        else:
            print(f"  MISSING  {match_id}.pt")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Cannot build splits because the following "
            "processed matches are missing:\n"
            + "\n".join(str(p) for p in missing)
        )

    outputs = {}

    # ------------------------------------------------------------------
    # Build each split independently.
    # ------------------------------------------------------------------

    for split_name, match_ids in SPLITS.items():
        data = combine_matches(
            processed_dir,
            match_ids,
            split_name,
        )

        output_path = (
            processed_dir
            / f"{split_name}.pt"
        )

        print_summary(
            split_name,
            data,
        )

        print("\nSaving:", output_path)

        torch.save(
            data,
            output_path,
        )

        outputs[split_name] = output_path

        print(
            f"{split_name} split saved successfully."
        )

    # ------------------------------------------------------------------
    # Final cross-split checks.
    # ------------------------------------------------------------------

    train_ids = set(SPLITS["train"])
    val_ids = set(SPLITS["val"])
    test_ids = set(SPLITS["test"])

    if train_ids & val_ids:
        raise RuntimeError(
            "Train and validation match sets overlap"
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train and test match sets overlap"
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Validation and test match sets overlap"
        )

    print()
    print("=" * 70)
    print("FINAL SPLIT BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nTRAIN")
    print(" matches:", SPLITS["train"])
    print(" file:", outputs["train"])

    print("\nVALIDATION")
    print(" matches:", SPLITS["val"])
    print(" file:", outputs["val"])

    print("\nTEST")
    print(" matches:", SPLITS["test"])
    print(" file:", outputs["test"])

    print()
    print("No match appears in more than one split.")


if __name__ == "__main__":
    main()