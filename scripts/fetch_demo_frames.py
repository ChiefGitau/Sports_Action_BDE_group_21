"""Fetch the real camera frames behind the demo clips, without downloading the whole match video.

SoccerTrack v2 ships one 4096x1080 panorama video per half (3.5 GB each). The GSR annotations were
made on exactly these frames (file_name 000001.jpg == video frame 1, seq_length == frame count), so
window start_frame values index straight into the video. OpenCV's FFmpeg backend seeks over HTTPS
with range requests, so only the needed 10 s are transferred.

    python scripts/fetch_demo_frames.py                 # match 117093, 2nd half, frames 53077..53316

Writes data/raw/frames/<match>_<half>_<first>_<n>.mp4 (frames downscaled to 2048x540) and
<...>_boxes.json with the image-space player boxes for the same frames (used for the overlay and as
an alignment check). Needs a Hugging Face login (hf auth login) that has accepted the dataset terms."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import ijson
import requests
from huggingface_hub import get_token

REPO = Path(__file__).resolve().parents[1]
FIRST_FRAME, NFRAMES = 53077, 240        # start_frame of window 4423 in demo_117093_half2.pt, 20 windows x 12 frames

DATASET = "atomscott/soccertrack-v2"
OUT_DIR = REPO / "data" / "raw" / "frames"


def video_url(match: str, half: int) -> str:
    name = f"{match}_panorama_{'1st' if half == 1 else '2nd'}_half.mp4"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/videos/{match}/{name}"
    tok = get_token()
    if tok is None:
        raise SystemExit("no Hugging Face token found: run `hf auth login` and accept the dataset terms")
    # resolve the redirect once; the signed CDN URL needs no auth header, which FFmpeg cannot send
    r = requests.head(url, headers={"Authorization": f"Bearer {tok}"}, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.url


def fetch_frames(url: str, first: int, n: int, out: Path, width: int = 2048) -> None:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit("could not open the remote video with OpenCV/FFmpeg")
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"remote video: {total} frames @ {fps:g} fps, {w}x{h}")
    size = (width, round(h * width / w))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)              # file_name NNNNNN.jpg is 1-based
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    assert vw.isOpened(), "avc1 writer failed"
    t0 = time.time()
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            raise SystemExit(f"read failed at frame {first + i}")
        vw.write(cv2.resize(fr, size, interpolation=cv2.INTER_AREA))
    vw.release()
    print(f"wrote {out.name}: {n} frames from {first}, {size[0]}x{size[1]}, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s")


def fetch_boxes(gsr: Path, first: int, n: int, out: Path) -> None:
    """Image-space player boxes for frames first..first+n-1, streamed from the GSR json."""
    ids: dict[str, int] = {}
    with gsr.open("rb") as f:
        for im in ijson.items(f, "images.item"):
            fr = int(Path(im["file_name"]).stem)
            if first <= fr < first + n:
                ids[str(im["image_id"])] = fr
    boxes: dict[int, list] = {}
    t0 = time.time()
    with gsr.open("rb") as f:
        for a in ijson.items(f, "annotations.item"):
            if a.get("supercategory") != "object":
                continue
            fr = ids.get(str(a["image_id"]))
            if fr is None:
                continue
            b, at = a["bbox_image"], a.get("attributes") or {}
            boxes.setdefault(fr, []).append({"track_id": a.get("track_id"), "team": at.get("team"), "role": at.get("role"),
                                             "x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"])})
    out.write_text(json.dumps(boxes))
    print(f"wrote {out.name}: boxes for {len(boxes)} frames, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="117093")
    ap.add_argument("--half", type=int, default=2)
    ap.add_argument("--first", type=int, default=FIRST_FRAME, help="1-based first video frame")
    ap.add_argument("--n", type=int, default=NFRAMES)
    ap.add_argument("--skip-boxes", action="store_true", help="skip the (slow) scan of the 5 GB GSR json")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{a.match}_{'1st' if a.half == 1 else '2nd'}_{a.first}_{a.n}"
    fetch_frames(video_url(a.match, a.half), a.first, a.n, stem.with_suffix(".mp4"))
    if not a.skip_boxes:
        gsr = REPO / "data" / "raw" / "gsr" / a.match / f"{a.match}_{'1st' if a.half == 1 else '2nd'}.json"
        fetch_boxes(gsr, a.first, a.n, Path(str(stem) + "_boxes.json"))
