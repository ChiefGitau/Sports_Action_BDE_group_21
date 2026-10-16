"""Render two 10 s clips (25 fps, 1280x720, H.264) of the model running on 20 consecutive
windows: one with the early-exit CNN and the adaptive policy, one with the full static CNN.

    python scripts/fetch_demo_frames.py            # once: the real camera frames for these windows
    python scripts/render_demo_clips.py            # writes outputs/demo/clip_*.mp4 and still_*.png

Needs data/processed/demo_117093_half2.pt (any split in the data contract works if START/NWIN fit)
and data/raw/frames/<match>_<half>_<first_frame>_<n>.mp4 from fetch_demo_frames.py: the SoccerTrack
panorama frames the tracking data was annotated on, shown at the top with the tracked boxes.
Same windows, same camera and pitch animation, so the right-hand panel is the only difference."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core import load_flops, load_split, load_thresholds            # noqa: E402
from models.cnn import featurize                                     # noqa: E402
from models.registry import load_checkpoint                          # noqa: E402
from scripts.demo_runtime import stream_window                       # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
START, NWIN, FPS, T = 4423, 20, 25, 12
W, H = 1280, 720
FRAMES_DIR = REPO / "data" / "raw" / "frames"

# palette (matches the deck)
BG, PANEL, LINE = (16, 39, 27), (23, 54, 38), (63, 90, 75)
LIGHT, SOFT, MUTED = (244, 245, 239), (195, 209, 199), (143, 165, 150)
GREEN, AMBER, BLUE, RED = (47, 143, 91), (227, 169, 47), (127, 179, 230), (214, 92, 76)
PITCH = (30, 90, 58)


def font(size: int, bold: bool = False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in cands:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


F_TITLE, F_H, F_B, F_S, F_BIG = font(30, True), font(24, True), font(22), font(18), font(40, True)


# ------------------------------------------------------------------ model runs
torch.set_num_threads(1)
split = load_split(str(REPO / "data/processed/demo_117093_half2.pt"))
X, M, y = split["X"], split["mask"], split["y"]
frames0 = split["meta"]["start_frame"]

# camera frames (see fetch_demo_frames.py): one video frame per (window, frame) pair, plus tracked boxes
FIRST_FRAME = int(frames0[START])
_stem = f"{split['meta']['match_id'][START]}_{'1st' if int(split['meta']['half'][START]) == 1 else '2nd'}_{FIRST_FRAME}_{NWIN * T}"
_cam_path, _box_path = FRAMES_DIR / f"{_stem}.mp4", FRAMES_DIR / f"{_stem}_boxes.json"
if not _cam_path.exists():
    raise SystemExit(f"missing {_cam_path}\nrun: python scripts/fetch_demo_frames.py --first {FIRST_FRAME} --n {NWIN * T}")
CX0, CY0, CW = 40, 100, W - 80                                   # camera strip on the canvas
SRC_W, SRC_H, SRC_Y0, SRC_Y1 = 4096, 1080, 110, 770              # native panorama size; rows kept (the pitch band)
CH = round(CW * (SRC_Y1 - SRC_Y0) / SRC_W)
_cap = cv2.VideoCapture(str(_cam_path))
cam_frames = []
while True:
    ok, fr = _cap.read()
    if not ok:
        break
    sy = fr.shape[0] / SRC_H
    band = fr[round(SRC_Y0 * sy) : round(SRC_Y1 * sy)]
    cam_frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(band, (CW, CH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
_cap.release()
assert len(cam_frames) == NWIN * T, f"{_cam_path.name}: {len(cam_frames)} frames, expected {NWIN * T}"
cam_boxes = {int(k): v for k, v in json.loads(_box_path.read_text()).items()} if _box_path.exists() else {}

ee, _ = load_checkpoint(str(REPO / "outputs/earlyexit/model.pt"))
st, _ = load_checkpoint(str(REPO / "outputs/cnn_full/model.pt"))
ee_flops = [int(f) for f in load_flops(str(REPO / "outputs/earlyexit/flops.json"))["flops_per_exit"]]
st_flops = int(load_flops(str(REPO / "outputs/cnn_full/flops.json"))["flops_per_exit"][0])
thr = [float(t) for t in load_thresholds(str(REPO / "outputs/earlyexit/thresholds.json"))["100"]]


@torch.no_grad()
def run_static(x, m):
    t0 = time.perf_counter()
    p = st(x, m)[0].softmax(-1)
    return {"exit": 0, "pred": int(p.argmax()), "conf": float(p.max()), "ms": (time.perf_counter() - t0) * 1000}


for _ in range(5):                       # warm-up
    stream_window(ee, X[:1], M[:1], thr)
    run_static(X[:1], M[:1])

results = {"ee": [], "st": []}
for i in range(START, START + NWIN):
    results["ee"].append(stream_window(ee, X[i : i + 1], M[i : i + 1], thr))
    results["st"].append(run_static(X[i : i + 1], M[i : i + 1]))


# ------------------------------------------------------------------ drawing helpers
PX0, PY0, PW, PH = 40, 322, 360, 233          # pitch box (105 x 68 m), under the camera strip


def pitch_xy(x, y):
    return PX0 + float(x) * PW, PY0 + float(y) * PH


def draw_pitch(d: ImageDraw.ImageDraw):
    d.rectangle([PX0, PY0, PX0 + PW, PY0 + PH], fill=PITCH, outline=LIGHT, width=2)
    cx = PX0 + PW / 2
    d.line([cx, PY0, cx, PY0 + PH], fill=LIGHT, width=2)
    d.ellipse([cx - 61, PY0 + PH / 2 - 61, cx + 61, PY0 + PH / 2 + 61], outline=LIGHT, width=2)
    s = PW / 105
    for side in (0, 1):
        bx = PX0 if side == 0 else PX0 + PW - 16.5 * s
        d.rectangle([bx, PY0 + (68 - 40.3) / 2 * s, bx + 16.5 * s, PY0 + (68 + 40.3) / 2 * s], outline=LIGHT, width=2)
        gx = PX0 if side == 0 else PX0 + PW - 5.5 * s
        d.rectangle([gx, PY0 + (68 - 18.3) / 2 * s, gx + 5.5 * s, PY0 + (68 + 18.3) / 2 * s], outline=LIGHT, width=2)


def text(d, xy, s, f=F_B, fill=LIGHT, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def box(d, xy, fill=None, outline=LINE, width=2, radius=10):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def fmt_mflops(v):
    return f"{v / 1e6:.2f} M"


# ------------------------------------------------------------------ frame renderer
def render(kind: str, w: int, f: int, cum_before: int) -> Image.Image:
    """kind 'ee' or 'st'; window w (0..NWIN-1); frame f (0..T-1) inside it."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    r = results[kind][w]
    gi = START + w
    label = int(y[gi])
    tsec = float(frames0[gi]) / FPS

    # titles
    if kind == "ee":
        text(d, (40, 28), "Early-exit CNN  ·  adaptive policy, budget 100%", F_TITLE, LIGHT)
    else:
        text(d, (40, 28), "Full CNN  ·  static baseline, always full depth", F_TITLE, LIGHT)
    text(d, (40, 70), f"SoccerTrack v2, match 117093, 2nd half  ·  window {gi}  ·  {int(tsec // 60):02d}:{tsec % 60:04.1f}", F_S, MUTED)

    # camera strip: the real panorama frame behind this window frame, with the tracked boxes
    frame_no = FIRST_FRAME + w * T + f
    img.paste(cam_frames[w * T + f], (CX0, CY0))
    sx, sy = CW / SRC_W, CH / (SRC_Y1 - SRC_Y0)
    for b in cam_boxes.get(frame_no, []):
        col = AMBER if b["team"] == "right" else (BLUE if b["team"] == "left" else LIGHT)
        bx0, by0 = CX0 + b["x"] * sx, CY0 + (b["y"] - SRC_Y0) * sy
        d.rectangle([bx0, by0, bx0 + b["w"] * sx, by0 + b["h"] * sy], outline=col, width=1)
    d.rectangle([CX0, CY0, CX0 + CW, CY0 + CH], outline=LINE, width=1)
    cap = f"camera panorama, frame {frame_no}   ·   tracked boxes: amber = right team, blue = left team, white = referee"
    tw = d.textlength(cap, font=F_S)
    d.rectangle([CX0 + 1, CY0 + CH - 28, CX0 + tw + 17, CY0 + CH - 1], fill=BG)
    text(d, (CX0 + 9, CY0 + CH - 8), cap, F_S, LIGHT, "ls")

    # pitch + players for this frame
    draw_pitch(d)
    xf, mf = X[gi, f], M[gi, f]
    for p in range(23):
        if not bool(mf[p]):
            continue
        px, py = pitch_xy(xf[p, 0], xf[p, 1])
        col = AMBER if p < 11 else (BLUE if p < 22 else LIGHT)
        rad = 7 if p < 22 else 5
        d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col, outline=BG, width=1)
    # label pill under pitch
    ly = PY0 + PH + 10                                   # label pill + legend under the pitch
    if label == 1:
        box(d, (PX0, ly, PX0 + 170, ly + 32), fill=AMBER, outline=AMBER)
        text(d, (PX0 + 85, ly + 16), "ACTION in window", F_S, BG, "mm")
    else:
        box(d, (PX0, ly, PX0 + 170, ly + 32), fill=None, outline=LINE)
        text(d, (PX0 + 85, ly + 16), "no action", F_S, SOFT, "mm")
    text(d, (PX0 + 186, ly + 16), "bird's-eye view of the\nsame tracking data", F_S, MUTED, "lm")

    # ---- right panel: trunk
    x0, y0 = 460, 322
    box(d, (x0 - 16, y0 - 12, W - 24, 600), fill=PANEL, outline=PANEL)
    text(d, (x0, y0), "Trunk executed for this window", F_H, SOFT)
    n_exec = (r["exit"] + 1) if kind == "ee" else 4
    decided_at = 2 * n_exec + 2                         # frame at which the decision appears
    for k in range(1, 5):
        by = y0 + 40 + (k - 1) * 46
        runs = k <= n_exec and f >= 2 * k
        pending = k <= n_exec and f < 2 * k
        skipped = k > n_exec
        fill = GREEN if runs else (PANEL if pending else None)
        outline = GREEN if (runs or pending) else LINE
        box(d, (x0, by, x0 + 130, by + 38), fill=fill, outline=outline)
        text(d, (x0 + 65, by + 19), f"Block {k}", F_B, LIGHT if runs else (SOFT if pending else MUTED), "mm")
        if skipped:
            text(d, (x0 + 150, by + 19), "not executed", F_S, MUTED, "lm")
            continue
        if kind == "ee" and runs:
            c = r["confidences"][k - 1]
            taken = (k - 1) == r["exit"]
            if taken and k == 4:
                text(d, (x0 + 150, by + 19), f"exit 4: conf {c:.3f}, final exit  →  STOP", F_S, AMBER, "lm")
            elif taken:
                text(d, (x0 + 150, by + 19), f"exit {k}: conf {c:.3f} ≥ {thr[k-1]:.2f}  →  STOP", F_S, AMBER, "lm")
            else:
                text(d, (x0 + 150, by + 19), f"exit {k}: conf {c:.3f} < {thr[k-1]:.2f}  →  continue", F_S, SOFT, "lm")
        elif kind == "st" and runs and k == 4:
            text(d, (x0 + 150, by + 19), f"single head: conf {r['conf']:.3f}", F_S, SOFT, "lm")

    # ---- decision
    dy = y0 + 40 + 4 * 46 + 6
    if f >= decided_at:
        pred = r["pred"]
        ok = pred == label
        col = GREEN if ok else RED
        box(d, (x0, dy, W - 40, dy + 40), fill=None, outline=col)
        text(d, (x0 + 14, dy + 20), f"prediction: {'ACTION' if pred else 'no action'}", F_H, col, "lm")
        text(d, (W - 54, dy + 20), "correct" if ok else "wrong", F_S, col, "rm")
        cost = ee_flops[r["exit"]] if kind == "ee" else st_flops
        share = cost / ee_flops[-1] if kind == "ee" else 1.0
        text(d, (W - 130, dy + 20), f"{cost:,} FLOPs ({100 * share:.0f}%)  ·  {r['ms']:.2f} ms", F_S, SOFT, "rm")
    else:
        box(d, (x0, dy, W - 40, dy + 40), fill=None, outline=LINE)
        text(d, (x0 + 14, dy + 20), "running ...", F_H, MUTED, "lm")

    # ---- bottom strip: cumulative compute + timeline
    sy = 608
    cost_now = (ee_flops[r["exit"]] if kind == "ee" else st_flops) if f >= decided_at else 0
    cum = cum_before + cost_now
    ref = NWIN * ee_flops[-1]
    text(d, (40, sy), f"compute so far: {fmt_mflops(cum)}FLOPs", F_B, LIGHT)
    text(d, (W - 40, sy), f"full depth on all 20 windows: {fmt_mflops(ref)}FLOPs", F_S, MUTED, "ra")
    bx0, bx1, byy = 40, W - 40, sy + 32
    d.rectangle([bx0, byy, bx1, byy + 16], fill=PANEL, outline=LINE)
    fillw = int((bx1 - bx0) * min(1.0, cum / ref))
    if fillw > 0:
        d.rectangle([bx0, byy, bx0 + fillw, byy + 16], fill=GREEN if kind == "ee" else AMBER)
    # timeline squares
    ty = byy + 26
    sq = (bx1 - bx0 - (NWIN - 1) * 6) / NWIN
    for j in range(NWIN):
        qx = bx0 + j * (sq + 6)
        done = j < w or (j == w and f >= decided_at)
        if done:
            e = results[kind][j]["exit"] if kind == "ee" else 3
            shade = [(47, 143, 91), (95, 170, 120), (160, 200, 170), (227, 169, 47)][e] if kind == "ee" else AMBER
            d.rectangle([qx, ty, qx + sq, ty + 12], fill=shade)
        else:
            d.rectangle([qx, ty, qx + sq, ty + 12], fill=None, outline=LINE)
        if int(y[START + j]) == 1:
            d.polygon([(qx + sq / 2 - 5, ty + 23), (qx + sq / 2 + 5, ty + 23), (qx + sq / 2, ty + 16)], fill=AMBER)
    if kind == "ee":
        text(d, (40, ty + 26), "timeline: dark green = exit 1, lighter = exit 2 or 3, amber = exit 4   ·   marker below = action window", F_S, MUTED)
    else:
        text(d, (40, ty + 26), "timeline: every window runs all four blocks   ·   marker below = action window", F_S, MUTED)
    return img


def render_clip(kind: str, path: Path, still: Path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    assert vw.isOpened(), "avc1 writer failed"
    cum = 0
    nframes = 0
    poster = None
    for w in range(NWIN):
        for f in range(T):
            img = render(kind, w, f, cum)
            if w == 8 and f == T - 1:
                poster = img
            vw.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            nframes += 1
        cum += ee_flops[results["ee"][w]["exit"]] if kind == "ee" else st_flops
    last = render(kind, NWIN - 1, T - 1, cum - (ee_flops[results["ee"][-1]["exit"]] if kind == "ee" else st_flops))
    for _ in range(FPS * 10 - nframes):              # hold the final state to exactly 10 s
        vw.write(cv2.cvtColor(np.asarray(last), cv2.COLOR_RGB2BGR))
        nframes += 1
    vw.release()
    poster.save(still)
    print(kind, path.name, f"{nframes} frames", f"{path.stat().st_size / 1e6:.2f} MB", "mean FLOPs",
          f"{cum / NWIN:,.0f}")


render_clip("ee", OUT / "clip_earlyexit.mp4", OUT / "still_earlyexit.png")
render_clip("st", OUT / "clip_fullcnn.mp4", OUT / "still_fullcnn.png")
for kind in ("ee", "st"):
    ex = [r["exit"] + 1 for r in results[kind]] if kind == "ee" else [4] * NWIN
    ok = sum(int(r["pred"] == int(y[START + j])) for j, r in enumerate(results[kind]))
    print(kind, "exits", ex, "correct", ok, "/", NWIN)
