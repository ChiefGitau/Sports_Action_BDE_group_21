from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import ijson
import torch


# ============================================================
# Repository imports
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, stride_frames, window_frames
from core.data import save_split


# ============================================================
# BAS EVENT PROCESSING
# ============================================================

def get_event_half(action: dict) -> str:
    """
    Extract the half number from BAS gameTime strings.

    Example:
        "1 - 0:03" -> "1"
        "2 - 7:42" -> "2"
    """
    game_time = str(action.get("gameTime", ""))
    return game_time.split("-", 1)[0].strip()


def load_bas_events(
    path: Path,
    half: int,
    seq_length: int,
) -> list[dict]:
    """
    Load BAS events belonging to the requested half.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    actions = data["actions"]

    selected = []

    for action in actions:

        if get_event_half(action) != str(half):
            continue

        frame = int(action["frame"])

        if not 1 <= frame <= seq_length:
            continue

        selected.append(action)

    return selected


def build_window_labels(
    actions: list[dict],
    seq_length: int,
    window_size: int,
    stride: int,
) -> list[dict]:
    """
    Construct temporal windows.

    y = 1 if at least one BAS action occurs in the window.
    y = 0 otherwise.
    """

    event_frames = sorted(
        {int(action["frame"]) for action in actions}
    )

    frame_to_labels = defaultdict(list)

    for action in actions:
        frame = int(action["frame"])
        frame_to_labels[frame].append(action["label"])

    windows = []

    for start in range(
        1,
        seq_length - window_size + 2,
        stride,
    ):

        end = start + window_size - 1

        idx = bisect_left(event_frames, start)

        positive = (
            idx < len(event_frames)
            and event_frames[idx] <= end
        )

        labels_in_window = []

        if positive:

            j = idx

            while (
                j < len(event_frames)
                and event_frames[j] <= end
            ):

                labels_in_window.extend(
                    frame_to_labels[event_frames[j]]
                )

                j += 1

        windows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "y": int(positive),
                "event_labels": labels_in_window,
            }
        )

    return windows


# ============================================================
# GSR IMAGE MAPPING
# ============================================================

def build_image_frame_map(
    gsr_path: Path,
) -> dict[str, int]:
    """
    Convert SoccerTrack image_id values into ordinary
    1-based frame numbers.

    Example:

        image_id = 3901000032
        file_name = 000032.jpg

    becomes:

        "3901000032" -> 32
    """

    mapping = {}

    with gsr_path.open("rb") as f:

        for image in ijson.items(
            f,
            "images.item",
        ):

            image_id = str(image["image_id"])

            frame = int(
                Path(image["file_name"]).stem
            )

            mapping[image_id] = frame

    return mapping


# ============================================================
# STABLE PLAYER SLOT ASSIGNMENT
# ============================================================

class StableSlotAllocator:
    """
    Keep players in stable tensor slots as much as possible.

    Tensor layout:

        slots 0..10   -> right team
        slots 11..21  -> left team
        slot 22       -> referee / unused

    SoccerTrack track IDs remain stable during a half, but a
    substitution may introduce a new track ID.

    When that happens, a slot belonging to an absent player can
    be reused without shifting all the other players.
    """

    def __init__(self, max_people: int):

        if max_people < 23:
            raise ValueError(
                "This slot layout requires max_people >= 23"
            )

        self.team_slots = {
            "right": list(range(0, 11)),
            "left": list(range(11, 22)),
        }

        self.track_to_slot: dict[str, dict[str, int]] = {
            "right": {},
            "left": {},
        }

        self.slot_to_track: dict[int, str] = {}

        self.last_seen: dict[str, int] = {}


    def assign_team(
        self,
        team: str,
        records: list[dict],
        frame: int,
    ) -> dict[str, int]:
        """
        Return track_id -> tensor slot for one team
        in the current frame.
        """

        allowed_slots = self.team_slots[team]

        # Deduplicate by track_id just in case.
        current_tracks = {
            str(record["track_id"])
            for record in records
        }

        if len(current_tracks) > len(allowed_slots):
            raise RuntimeError(
                f"Frame {frame}: team {team} has "
                f"{len(current_tracks)} tracks but only "
                f"{len(allowed_slots)} available slots"
            )

        mapping = self.track_to_slot[team]

        result = {}

        # ----------------------------------------------------
        # First preserve players that already have slots.
        # ----------------------------------------------------

        for track_id in sorted(current_tracks):

            if track_id in mapping:

                slot = mapping[track_id]

                result[track_id] = slot
                self.last_seen[track_id] = frame

        # Slots already required by currently-present tracks.
        active_slots = set(result.values())

        # ----------------------------------------------------
        # Assign slots to newly appearing tracks.
        # ----------------------------------------------------

        new_tracks = sorted(
            track_id
            for track_id in current_tracks
            if track_id not in mapping
        )

        for track_id in new_tracks:

            candidates = [
                slot
                for slot in allowed_slots
                if slot not in active_slots
            ]

            if not candidates:
                raise RuntimeError(
                    f"Frame {frame}: no free {team} slot "
                    f"for track {track_id}"
                )

            # Prefer a never-used slot.
            never_used = [
                slot
                for slot in candidates
                if slot not in self.slot_to_track
            ]

            if never_used:

                slot = min(never_used)

            else:

                # Otherwise recycle the slot whose old owner
                # has been absent for the longest time.
                def old_owner_last_seen(slot_number: int):
                    old_track = self.slot_to_track[slot_number]
                    return self.last_seen.get(old_track, -1)

                slot = min(
                    candidates,
                    key=old_owner_last_seen,
                )

                old_track = self.slot_to_track.get(slot)

                if old_track is not None:
                    mapping.pop(old_track, None)

            mapping[track_id] = slot
            self.slot_to_track[slot] = track_id
            self.last_seen[track_id] = frame

            result[track_id] = slot
            active_slots.add(slot)

        return result


# ============================================================
# FRAME -> WINDOW LOOKUP
# ============================================================

def build_frame_targets(
    windows: list[dict],
    seq_length: int,
) -> list[list[tuple[int, int]]]:
    """
    For every frame, record which dataset window(s) use it.

    Each element contains:

        (window_index, time_index_inside_window)
    """

    targets = [
        []
        for _ in range(seq_length + 1)
    ]

    for window_index, window in enumerate(windows):

        start = window["start_frame"]
        end = window["end_frame"]

        for frame in range(start, end + 1):

            time_index = frame - start

            targets[frame].append(
                (
                    window_index,
                    time_index,
                )
            )

    return targets


# ============================================================
# GSR FRAME PROCESSING
# ============================================================

def process_frame(
    frame: int,
    records: list[dict],
    allocator: StableSlotAllocator,
    frame_targets: list[list[tuple[int, int]]],
    X: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Convert one SoccerTrack frame into tensor entries.
    """

    if frame <= 0 or frame >= len(frame_targets):
        return

    targets = frame_targets[frame]

    # Tail frames that do not belong to a complete window
    # require no tensor work.
    if not targets:
        return

    team_records = {
        "right": [],
        "left": [],
    }

    referee_records = []

    # --------------------------------------------------------
    # Filter useful person annotations
    # --------------------------------------------------------

    for record in records:

        attrs = record.get("attributes") or {}

        role = attrs.get("role")
        team = attrs.get("team")

        if role in {"player", "goalkeeper"}:

            if team in {"left", "right"}:
                team_records[team].append(record)

        elif role == "referee":

            referee_records.append(record)

    # --------------------------------------------------------
    # Stable team slot assignment
    # --------------------------------------------------------

    slot_maps = {}

    for team in ("right", "left"):

        slot_maps[team] = allocator.assign_team(
            team,
            team_records[team],
            frame,
        )

    # --------------------------------------------------------
    # Write team-player positions
    # --------------------------------------------------------

    for team in ("right", "left"):

        for record in team_records[team]:

            track_id = str(record["track_id"])

            slot = slot_maps[team][track_id]

            pitch = record.get("bbox_pitch") or {}

            if (
                "x_bottom_middle" not in pitch
                or "y_bottom_middle" not in pitch
            ):
                continue

            x = float(
                pitch["x_bottom_middle"]
            )

            y = float(
                pitch["y_bottom_middle"]
            )

            # Fixed 105 x 68 metre football pitch.
            #
            # Values are intentionally NOT clipped because
            # SoccerTrack occasionally contains coordinates
            # slightly outside the nominal pitch boundaries.
            x_norm = (x + 52.5) / 105.0
            y_norm = (y + 34.0) / 68.0

            for window_index, time_index in targets:

                X[
                    window_index,
                    time_index,
                    slot,
                    0,
                ] = x_norm

                X[
                    window_index,
                    time_index,
                    slot,
                    1,
                ] = y_norm

                mask[
                    window_index,
                    time_index,
                    slot,
                ] = True

    # --------------------------------------------------------
    # Optional referee slot
    # --------------------------------------------------------

    if referee_records:

        record = referee_records[0]

        pitch = record.get("bbox_pitch") or {}

        if (
            "x_bottom_middle" in pitch
            and
            "y_bottom_middle" in pitch
        ):

            x = float(
                pitch["x_bottom_middle"]
            )

            y = float(
                pitch["y_bottom_middle"]
            )

            x_norm = (x + 52.5) / 105.0
            y_norm = (y + 34.0) / 68.0

            referee_slot = 22

            for window_index, time_index in targets:

                X[
                    window_index,
                    time_index,
                    referee_slot,
                    0,
                ] = x_norm

                X[
                    window_index,
                    time_index,
                    referee_slot,
                    1,
                ] = y_norm

                mask[
                    window_index,
                    time_index,
                    referee_slot,
                ] = True


# ============================================================
# REAL GSR -> TENSORS
# ============================================================

def build_real_tensors(
    gsr_path: Path,
    image_to_frame: dict[str, int],
    windows: list[dict],
    seq_length: int,
    window_size: int,
    max_people: int,
):
    """
    Stream the GSR annotations and construct:

        X    [N, T, P, 2]
        mask [N, T, P]
    """

    N = len(windows)
    T = window_size
    P = max_people

    X = torch.zeros(
        (N, T, P, 2),
        dtype=torch.float32,
    )

    mask = torch.zeros(
        (N, T, P),
        dtype=torch.bool,
    )

    frame_targets = build_frame_targets(
        windows,
        seq_length,
    )

    allocator = StableSlotAllocator(
        max_people=P
    )

    current_frame = None
    current_records = []

    previous_frame = 0

    processed_frames = 0
    processed_records = 0

    print(
        "\nStreaming GSR annotations..."
    )

    with gsr_path.open("rb") as f:

        for ann in ijson.items(
            f,
            "annotations.item",
        ):

            if ann.get("supercategory") != "object":
                continue

            image_id = str(
                ann.get("image_id")
            )

            frame = image_to_frame.get(
                image_id
            )

            if frame is None:
                continue

            if frame > seq_length:
                continue

            # Ensure the stream is frame ordered.
            if frame < previous_frame:
                raise RuntimeError(
                    "GSR annotations are not ordered by frame. "
                    "The streaming builder requires ordered "
                    "annotations."
                )

            previous_frame = frame

            if current_frame is None:

                current_frame = frame

            if frame != current_frame:

                process_frame(
                    current_frame,
                    current_records,
                    allocator,
                    frame_targets,
                    X,
                    mask,
                )

                processed_frames += 1
                processed_records += len(
                    current_records
                )

                if processed_frames % 5000 == 0:
                    print(
                        "processed frames:",
                        processed_frames,
                    )

                current_frame = frame
                current_records = []

            current_records.append(ann)

    # Process final frame.
    if current_frame is not None:

        process_frame(
            current_frame,
            current_records,
            allocator,
            frame_targets,
            X,
            mask,
        )

        processed_frames += 1
        processed_records += len(
            current_records
        )

    print(
        "GSR frames processed:",
        processed_frames,
    )

    print(
        "object annotations processed:",
        processed_records,
    )

    return X, mask


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bas",
        type=Path,
        required=True,
        help="Path to BAS class-events JSON",
    )

    parser.add_argument(
        "--gsr",
        type=Path,
        required=True,
        help="Path to SoccerTrack GSR JSON",
    )

    parser.add_argument(
        "--half",
        type=int,
        required=True,
        choices=[1, 2],
        help="Match half to process",
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        required=True,
        help="Number of frames in this GSR half",
    )

    parser.add_argument(
        "--match-id",
        type=str,
        required=True,
        help="SoccerTrack match ID",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt path",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    cfg = load_config()

    T = window_frames(cfg)
    stride = stride_frames(cfg)
    P = int(cfg.data.max_people)

    print(
        "=== CONFIG ==="
    )

    print(
        "fps:",
        cfg.data.fps,
    )

    print(
        "window_seconds:",
        cfg.data.window_seconds,
    )

    print(
        "stride_seconds:",
        cfg.data.stride_seconds,
    )

    print(
        "window_frames:",
        T,
    )

    print(
        "stride_frames:",
        stride,
    )

    print(
        "max_people:",
        P,
    )

    print(
        "num_classes:",
        cfg.data.num_classes,
    )

    # --------------------------------------------------------
    # BAS
    # --------------------------------------------------------

    actions = load_bas_events(
        args.bas,
        args.half,
        args.seq_length,
    )

    windows = build_window_labels(
        actions,
        args.seq_length,
        T,
        stride,
    )

    positives = sum(
        window["y"]
        for window in windows
    )

    negatives = (
        len(windows)
        - positives
    )

    print(
        "\n=== BAS / WINDOWS ==="
    )

    print(
        "events:",
        len(actions),
    )

    print(
        "total windows:",
        len(windows),
    )

    print(
        "positive windows:",
        positives,
    )

    print(
        "negative windows:",
        negatives,
    )

    print(
        "positive fraction:",
        round(
            positives / len(windows),
            4,
        ),
    )

    # --------------------------------------------------------
    # GSR image map
    # --------------------------------------------------------

    print(
        "\nBuilding image_id -> frame mapping..."
    )

    image_to_frame = build_image_frame_map(
        args.gsr
    )

    print(
        "mapped images:",
        len(image_to_frame),
    )

    # --------------------------------------------------------
    # Real tensors
    # --------------------------------------------------------

    X, mask = build_real_tensors(
        gsr_path=args.gsr,
        image_to_frame=image_to_frame,
        windows=windows,
        seq_length=args.seq_length,
        window_size=T,
        max_people=P,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = torch.tensor(
        [
            window["y"]
            for window in windows
        ],
        dtype=torch.int64,
    )

    # --------------------------------------------------------
    # Metadata required by Core
    # --------------------------------------------------------

    N = len(windows)

    meta = {
        "match_id": [
            args.match_id
            for _ in range(N)
        ],

        "half": torch.full(
            (N,),
            args.half,
            dtype=torch.int64,
        ),

        "start_frame": torch.tensor(
            [
                window["start_frame"]
                for window in windows
            ],
            dtype=torch.int64,
        ),
    }

    dataset = {
        "X": X,
        "mask": mask,
        "y": y,
        "meta": meta,
    }

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        "\n=== FINAL TENSOR SHAPES ==="
    )

    print(
        "X:",
        tuple(X.shape),
        X.dtype,
    )

    print(
        "mask:",
        tuple(mask.shape),
        mask.dtype,
    )

    print(
        "y:",
        tuple(y.shape),
        y.dtype,
    )

    print(
        "\n=== OCCUPANCY ==="
    )

    present = int(
        mask.sum().item()
    )

    possible = mask.numel()

    print(
        "present person-slots:",
        present,
    )

    print(
        "possible person-slots:",
        possible,
    )

    print(
        "occupancy fraction:",
        round(
            present / possible,
            4,
        ),
    )

    print(
        "\n=== LABEL BALANCE ==="
    )

    print(
        "y=0:",
        int((y == 0).sum().item()),
    )

    print(
        "y=1:",
        int((y == 1).sum().item()),
    )

    # --------------------------------------------------------
    # Save through Core validator
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nValidating and saving..."
    )

    save_split(
        dataset,
        args.output,
        num_classes=int(
            cfg.data.num_classes
        ),
    )

    print(
        "saved:",
        args.output,
    )

    print(
        "\nM1 half dataset build completed successfully."
    )


if __name__ == "__main__":
    main()