from pathlib import Path
import ijson

path = Path(r"data\raw\117092\117092_1st.json")

target_frame = 32
target_filename = f"{target_frame:06d}.jpg"

target_image_id = None

# Find which internal image_id corresponds to frame 32.
with path.open("rb") as f:
    for image in ijson.items(f, "images.item"):
        if image.get("file_name") == target_filename:
            target_image_id = str(image["image_id"])
            print("Frame:", target_frame)
            print("Filename:", target_filename)
            print("GSR image_id:", target_image_id)
            break

if target_image_id is None:
    raise RuntimeError("Could not find target frame")

print("\nPlayers present at this frame:")

found_actor = False

with path.open("rb") as f:
    for ann in ijson.items(f, "annotations.item"):
        if ann.get("supercategory") != "object":
            continue

        if str(ann.get("image_id")) != target_image_id:
            continue

        attrs = ann.get("attributes") or {}

        print(
            "track_id=", ann.get("track_id"),
            "player_id=", attrs.get("player_id"),
            "team=", attrs.get("team"),
            "role=", attrs.get("role"),
        )

        if str(attrs.get("player_id")) == "467259":
            found_actor = True

print("\nBAS actor 467259 found:", found_actor)