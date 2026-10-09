from pathlib import Path
from collections import Counter
import argparse
import statistics

import ijson


def inspect_gsr(path: Path) -> None:
    print(f"Inspecting: {path}")
    print(f"File size: {path.stat().st_size / (1024**3):.2f} GB")

    # Read only the small metadata object first.
    with path.open("rb") as f:
        info = next(ijson.items(f, "info"))

    print("\n=== FILE INFO ===")
    print("version:", info.get("version"))
    print("frame_rate:", info.get("frame_rate"))
    print("seq_length:", info.get("seq_length"))

    print("\n=== FIRST 5 IMAGE RECORDS ===")

    with path.open("rb") as f:
        for i, image in enumerate(ijson.items(f, "images.item")):
            print(image)
            if i == 4:
                break

    frame_counts = Counter()
    roles = Counter()
    teams = Counter()

    x_values = []
    y_values = []

    samples = []
    object_count = 0

    # Stream through annotations one at a time.
    with path.open("rb") as f:
        for ann in ijson.items(f, "annotations.item"):

            # Ignore pitch-line and camera records.
            if ann.get("supercategory") != "object":
                continue

            object_count += 1

            image_id = str(ann.get("image_id"))

            attributes = ann.get("attributes") or {}
            role = attributes.get("role", "unknown")
            team = attributes.get("team")

            bbox_pitch = ann.get("bbox_pitch") or {}
            x = bbox_pitch.get("x_bottom_middle")
            y = bbox_pitch.get("y_bottom_middle")

            frame_counts[image_id] += 1
            roles[str(role)] += 1
            teams[str(team)] += 1

            if x is not None and y is not None:
                x_values.append(float(x))
                y_values.append(float(y))

            if len(samples) < 5:
                samples.append(
                    {
                        "image_id": image_id,
                        "track_id": ann.get("track_id"),
                        "player_id": attributes.get("player_id"),
                        "role": role,
                        "team": team,
                        "jersey": attributes.get("jersey"),
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                    }
                )

    print("\n=== OBJECT ANNOTATIONS ===")
    print("object records:", object_count)
    print("annotated frames:", len(frame_counts))

    if frame_counts:
        people_per_frame = list(frame_counts.values())

        print("\n=== PEOPLE PER FRAME ===")
        print("minimum:", min(people_per_frame))
        print("mean:", round(statistics.mean(people_per_frame), 2))
        print("median:", statistics.median(people_per_frame))
        print("maximum:", max(people_per_frame))

    print("\n=== ROLES ===")
    for role, count in roles.most_common():
        print(f"{role}: {count}")

    print("\n=== TEAMS ===")
    for team, count in teams.most_common():
        print(f"{team}: {count}")

    if x_values and y_values:
        print("\n=== PITCH COORDINATES ===")
        print(f"x range: {min(x_values):.3f} to {max(x_values):.3f}")
        print(f"y range: {min(y_values):.3f} to {max(y_values):.3f}")

    print("\n=== FIRST 5 OBJECT RECORDS ===")
    for sample in samples:
        print(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    inspect_gsr(args.path)