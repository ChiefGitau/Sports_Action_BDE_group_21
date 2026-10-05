from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import ijson


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

DATASET_REPO = "atomscott/soccertrack-v2"


def run_command(command: list[str]) -> None:
    """Run a command and stop immediately if it fails."""

    print("\n>", " ".join(command))

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def download_file(
    repo_file: str,
    local_root: Path,
    expected_path: Path,
) -> None:
    """
    Download one SoccerTrack file with the Hugging Face CLI.

    If the file already exists locally, do not download it again.
    """

    if expected_path.exists():
        print("Already downloaded:", expected_path)
        return

    hf_executable = shutil.which("hf")

    if hf_executable is None:
        raise RuntimeError(
            "The 'hf' command was not found. "
            "Activate the project virtual environment first."
        )

    local_root.mkdir(parents=True, exist_ok=True)

    print("\nDownloading:")
    print(" ", repo_file)

    run_command(
        [
            hf_executable,
            "download",
            DATASET_REPO,
            repo_file,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(local_root),
        ]
    )

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Download finished, but the expected file was not found:\n"
            f"{expected_path}"
        )


def read_seq_length(gsr_path: Path) -> int:
    """
    Read the seq_length field from a large GSR JSON file
    without loading the whole file into RAM.
    """

    print("\nReading sequence length from:")
    print(" ", gsr_path)

    with gsr_path.open("rb") as f:
        info = next(ijson.items(f, "info"))

    frame_rate = info.get("frame_rate")
    seq_length = int(info["seq_length"])
    version = info.get("version")

    print(" version:", version)
    print(" frame_rate:", frame_rate)
    print(" seq_length:", seq_length)

    return seq_length


def build_half(
    match_id: str,
    half: int,
    bas_path: Path,
    gsr_path: Path,
    seq_length: int,
    output_path: Path,
) -> None:
    """Run the existing build_dataset.py pipeline for one half."""

    print("\n" + "=" * 60)
    print(f"BUILDING MATCH {match_id} HALF {half}")
    print("=" * 60)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_dataset.py"),
        "--bas",
        str(bas_path),
        "--gsr",
        str(gsr_path),
        "--half",
        str(half),
        "--seq-length",
        str(seq_length),
        "--match-id",
        match_id,
        "--output",
        str(output_path),
    ]

    run_command(command)


def combine_halves(
    first_path: Path,
    second_path: Path,
    output_path: Path,
) -> None:
    """Combine the two processed halves into one match dataset."""

    print("\n" + "=" * 60)
    print("COMBINING HALVES")
    print("=" * 60)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "combine_match.py"),
        "--first",
        str(first_path),
        "--second",
        str(second_path),
        "--output",
        str(output_path),
    ]

    run_command(command)


def process_match(match_id: str) -> None:
    """Download, build, and combine one complete SoccerTrack match."""

    match_raw_root = RAW_ROOT / match_id

    bas_repo_file = (
        f"bas/{match_id}/{match_id}_12_class_events.json"
    )

    gsr1_repo_file = (
        f"gsr/{match_id}/{match_id}_1st.json"
    )

    gsr2_repo_file = (
        f"gsr/{match_id}/{match_id}_2nd.json"
    )

    bas_path = (
        match_raw_root
        / "bas"
        / match_id
        / f"{match_id}_12_class_events.json"
    )

    gsr1_path = (
        match_raw_root
        / "gsr"
        / match_id
        / f"{match_id}_1st.json"
    )

    gsr2_path = (
        match_raw_root
        / "gsr"
        / match_id
        / f"{match_id}_2nd.json"
    )

    half1_output = (
        PROCESSED_ROOT / f"{match_id}_half1.pt"
    )

    half2_output = (
        PROCESSED_ROOT / f"{match_id}_half2.pt"
    )

    final_output = (
        PROCESSED_ROOT / f"{match_id}.pt"
    )

    print("=" * 60)
    print("SOCCERTRACK MATCH PROCESSOR")
    print("=" * 60)
    print("match:", match_id)
    print("raw directory:", match_raw_root)
    print("final output:", final_output)

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== DOWNLOADING BAS ===")

    download_file(
        bas_repo_file,
        match_raw_root,
        bas_path,
    )

    print("\n=== DOWNLOADING GSR HALF 1 ===")

    download_file(
        gsr1_repo_file,
        match_raw_root,
        gsr1_path,
    )

    print("\n=== DOWNLOADING GSR HALF 2 ===")

    download_file(
        gsr2_repo_file,
        match_raw_root,
        gsr2_path,
    )

    print("\n=== READING HALF LENGTHS ===")

    seq_length_1 = read_seq_length(gsr1_path)
    seq_length_2 = read_seq_length(gsr2_path)

    print("\n=== MATCH INFORMATION ===")
    print("match:", match_id)
    print("half 1 frames:", seq_length_1)
    print("half 2 frames:", seq_length_2)

    build_half(
        match_id=match_id,
        half=1,
        bas_path=bas_path,
        gsr_path=gsr1_path,
        seq_length=seq_length_1,
        output_path=half1_output,
    )

    build_half(
        match_id=match_id,
        half=2,
        bas_path=bas_path,
        gsr_path=gsr2_path,
        seq_length=seq_length_2,
        output_path=half2_output,
    )

    combine_halves(
        first_path=half1_output,
        second_path=half2_output,
        output_path=final_output,
    )

    print("\n" + "=" * 60)
    print("MATCH PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("match:", match_id)
    print("half 1:", half1_output)
    print("half 2:", half2_output)
    print("combined:", final_output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and preprocess one complete "
            "SoccerTrack v2 match."
        )
    )

    parser.add_argument(
        "--match-id",
        required=True,
        help="SoccerTrack match ID, for example 117093",
    )

    args = parser.parse_args()

    process_match(args.match_id)


if __name__ == "__main__":
    main()