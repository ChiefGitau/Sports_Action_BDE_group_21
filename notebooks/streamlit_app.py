"""Streamlit demo: run the trained models on a SoccerTrack clip under different edge scenarios.

    streamlit run notebooks/streamlit_app.py

Pick a model (full CNN, pruned CNN, early-exit CNN, early-exit INT8) and a battery scenario
(fixed 100/50/20 %, a draining battery, or a custom threshold). The app renders the clip in the
same layout as outputs/demo/clip_*.mp4: camera panorama with tracked boxes, bird's-eye pitch,
which trunk blocks ran, the exit taken, and the compute spent so far.

The models read player *tracking* data (data contract X [N, 12, 23, 2]), not pixels. The video is
what you see; the model input is the tracking window annotated on exactly those frames. A clip is
matched to its windows by its file name <match>_<1st|2nd>_<first_frame>_<n>.mp4, which is the name
scripts/fetch_demo_frames.py writes. Any other video can be shown, but you then choose the
tracking windows yourself (the predictions are for those windows, not for the uploaded pixels).
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import ijson
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core import load_flops, load_split, load_thresholds   # noqa: E402
from models.registry import load_checkpoint                                 # noqa: E402
from scripts.demo_runtime import level_for, stream_window                   # noqa: E402

T, FPS = 12, 25
W, H = 1280, 720
OUT = REPO / "outputs"
FRAMES_DIR = REPO / "data" / "raw" / "frames"
DEFAULT_VIDEO = FRAMES_DIR / "117093_2nd_53077_240.mp4"
DEFAULT_SPLIT = REPO / "data" / "processed" / "demo_117093_half2.pt"
MAX_WINDOWS = 50                                   # 600 video frames; keeps memory and render time sane
LEVELS = ("100", "50", "20")
# canonical match split (README section 4, scripts/build_splits.py)
SPLIT_OF_MATCH = {**{m: "training" for m in ("117092", "118575", "118576", "118577", "118578", "128058")},
                  **{m: "validation" for m in ("117093", "132877")}, **{m: "test" for m in ("128057", "132831")}}

# camera strip and pitch geometry (same as scripts/render_demo_clips.py)
SRC_W, SRC_H, SRC_Y0, SRC_Y1 = 4096, 1080, 110, 770     # native panorama size; rows kept (the pitch band)
CX0, CY0, CW = 40, 100, W - 80
CH = round(CW * (SRC_Y1 - SRC_Y0) / SRC_W)
PX0, PY0, PW, PH = 40, 322, 360, 233

# palette (matches the deck and the demo clips)
BG, PANEL, LINE = (16, 39, 27), (23, 54, 38), (63, 90, 75)
LIGHT, SOFT, MUTED = (244, 245, 239), (195, 209, 199), (143, 165, 150)
GREEN, AMBER, BLUE, RED = (47, 143, 91), (227, 169, 47), (127, 179, 230), (214, 92, 76)
PITCH = (30, 90, 58)
EXIT_SHADES = [(47, 143, 91), (95, 170, 120), (160, 200, 170), (227, 169, 47)]

MODELS = {
    "Full CNN": dict(key="full", adaptive=False, ckpt=OUT / "cnn_full/model.pt", flops=OUT / "cnn_full/flops.json",
                     blurb="static baseline, always all four blocks"),
    "Pruned CNN": dict(key="pruned", adaptive=False, ckpt=OUT / "cnn_pruned/model.pt", flops=OUT / "cnn_pruned/flops.json",
                       blurb="static, 50% of the channels physically removed"),
    "Early-exit CNN": dict(key="ee", adaptive=True, ckpt=OUT / "earlyexit/model.pt", flops=OUT / "earlyexit/flops.json",
                           thr=OUT / "earlyexit/thresholds.json", blurb="stops at the first confident exit"),
    "Early-exit CNN · INT8": dict(key="int8", adaptive=True, ckpt=OUT / "earlyexit_int8/earlyexit_int8.pt",
                                  flops=OUT / "earlyexit/flops.json", thr=OUT / "earlyexit_int8/thresholds.json",
                                  blurb="same policy, INT8 weights (post-training quantization)"),
}


# ------------------------------------------------------------------ fonts and drawing helpers

def font(size: int, bold: bool = False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


F_TITLE, F_H, F_B, F_S = font(30, True), font(24, True), font(22), font(18)


def text(d, xy, s, f=F_B, fill=LIGHT, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def box(d, xy, fill=None, outline=LINE, width=2, radius=10):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_pitch(d):
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


# ------------------------------------------------------------------ loading (cached)

@st.cache_resource(show_spinner="Loading models ...")
def load_models() -> dict:
    torch.set_num_threads(1)                     # the edge scenario: one window, one core
    engines = torch.backends.quantized.supported_engines
    for eng in ("onednn", "x86", "fbgemm", "qnnpack"):
        if eng in engines:
            torch.backends.quantized.engine = eng
            break
    full_ref = int(load_flops(OUT / "cnn_full/flops.json")["flops_per_exit"][0])
    loaded = {}
    for name, spec in MODELS.items():
        prof = load_flops(spec["flops"])
        if spec["key"] == "int8":
            model = torch.jit.load(str(spec["ckpt"]), map_location="cpu").eval()
            size_mb = _int8_size_mb()
        else:
            model, _ = load_checkpoint(spec["ckpt"])
            size_mb = float(prof["size_mb"])
        loaded[name] = dict(
            spec, model=model, flops=[int(f) for f in prof["flops_per_exit"]], size_mb=size_mb,
            thresholds=load_thresholds(spec["thr"]) if spec["adaptive"] else None,
            widths=model.channel_widths() if hasattr(model, "channel_widths") else None,
        )
    return {"models": loaded, "full_ref": full_ref}


def _int8_size_mb() -> float:
    """Size of the INT8 state_dict as reported by M6 (the TorchScript file also stores the graph)."""
    report = (OUT / "earlyexit_int8/quantization_report.txt").read_text()
    sizes = re.findall(r"size_mb:\s*([0-9.]+)", report)
    return float(sizes[-1]) if sizes else float("nan")


@st.cache_data(show_spinner=False)
def window_length(path: str, mtime: float) -> int | None:
    """T of a split without reading it all (memory-mapped). mtime only invalidates the cache."""
    try:
        return int(torch.load(path, mmap=True, map_location="cpu")["X"].shape[1])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def split_halves(path: str, mtime: float) -> set:
    """{(match_id, half)} covered by a split (memory-mapped). mtime only invalidates the cache."""
    meta = torch.load(path, mmap=True, map_location="cpu")["meta"]
    return {(str(mid), int(h)) for mid, h in zip(meta["match_id"], meta["half"].tolist())}


@st.cache_resource(show_spinner="Loading tracking split ...")
def load_tracking(path: str) -> dict:
    return load_split(path)


@st.cache_data(show_spinner=False)
def native_size(match: str, half: int, file_w: int, file_h: int) -> tuple[int, int]:
    """Native camera resolution of a SoccerTrack half; the GSR boxes are in these pixels.

    The cameras differ per match (117093: 4096x1080 panorama, 117092: 3840x1906 fisheye), and the
    clips are downscaled, so read it from the GSR json when present, else guess from the aspect."""
    gsr = REPO / "data" / "raw" / "gsr" / match / f"{match}_{'1st' if half == 1 else '2nd'}.json"
    if gsr.exists():
        with gsr.open("rb") as f:
            im = next(ijson.items(f, "images.item"))
        return int(im["width"]), int(im["height"])
    w = SRC_W if file_w / file_h > 3.0 else 3840
    return w, round(w * file_h / file_w)


def crop_geometry(native: tuple[int, int], boxes: dict) -> dict:
    """Rows of the native frame shown in the strip: full width, a band with the strip's aspect ratio,
    centred on where the tracked players are in this clip (the 4096x1080 panorama keeps the demo crop)."""
    w, h = native
    band = min(h, round(w * CH / CW))
    if (w, h) == (SRC_W, SRC_H):
        y0 = SRC_Y0
    else:
        ys = [b["y"] for v in boxes.values() for b in v] + [b["y"] + b["h"] for v in boxes.values() for b in v]
        centre = (min(ys) + max(ys)) / 2 if ys else h / 2
        y0 = round(min(max(centre - band / 2, 0), h - band))
    return {"w": w, "h": h, "y0": y0, "band": band}


@st.cache_data(show_spinner="Decoding video ...", max_entries=4)
def load_camera(path: str, digest: str, max_frames: int, geom: dict | None) -> dict:
    """Decode up to max_frames frames, cropped/resized to the camera strip (RGB uint8 arrays).

    geom (from crop_geometry) is set for SoccerTrack clips; other videos are letterboxed."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        h0, w0 = fr.shape[:2]
        if geom:
            sy = h0 / geom["h"]
            band = fr[round(geom["y0"] * sy): round((geom["y0"] + geom["band"]) * sy)]
            im = cv2.resize(band, (CW, CH), interpolation=cv2.INTER_AREA)
        else:                                       # any other video: letterbox into the strip
            scale = min(CW / w0, CH / h0)
            nw, nh = int(w0 * scale), int(h0 * scale)
            im = np.zeros((CH, CW, 3), np.uint8)
            im[:] = BG[::-1]
            y0, x0 = (CH - nh) // 2, (CW - nw) // 2
            im[y0:y0 + nh, x0:x0 + nw] = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    return {"frames": frames, "geom": geom}


def file_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def parse_clip_name(name: str):
    """'117093_2nd_53077_240.mp4' -> ('117093', 2, 53077). None if the name has another form."""
    m = re.match(r"(\d+)_(1st|2nd)_(\d+)_(\d+)", Path(name).stem)
    if not m:
        return None
    return m.group(1), 1 if m.group(2) == "1st" else 2, int(m.group(3))


def clip_label(path: Path) -> str:
    """'117093_2nd_53077_240.mp4' -> 'match 117093 · 2nd half · 35:23 – 35:32'."""
    parsed = parse_clip_name(path.name)
    if not parsed:
        return path.name
    match, half, first = parsed
    n = int(Path(path).stem.split("_")[-1])
    t0, t1 = (first - 1) / FPS, (first - 1 + n) / FPS
    role = f" ({SPLIT_OF_MATCH[match]})" if match in SPLIT_OF_MATCH else ""
    return (f"match {match}{role} · {'1st' if half == 1 else '2nd'} half · "
            f"{int(t0 // 60):02d}:{int(t0 % 60):02d} – {int(t1 // 60):02d}:{int(t1 % 60):02d}")


def find_window(split: dict, match: str, half: int, first_frame: int):
    """Index of the window starting at first_frame (exact), else the nearest earlier one + offset."""
    meta = split["meta"]
    sel = [i for i, mid in enumerate(meta["match_id"]) if str(mid) == match and int(meta["half"][i]) == half]
    if not sel:
        return None, None
    starts = meta["start_frame"][sel]
    before = (starts <= first_frame).nonzero().flatten()
    if len(before) == 0:
        return None, None
    j = int(before[-1])
    return sel[j], first_frame - int(starts[j])


# ------------------------------------------------------------------ inference and scenario simulation

@torch.no_grad()
def run_window(m: dict, x, mask, thr, repeats: int = 5) -> dict:
    """One window [1, T, P, 2]. Latency is the median of `repeats` runs (single runs are noisy)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if m["key"] == "ee":                         # truly incremental: later blocks never run
            r = stream_window(m["model"], x, mask, thr)
        elif m["key"] == "int8":                     # traced graph computes every exit; the policy picks one
            outs = m["model"](x, mask.float())
            probs = [o.softmax(-1) for o in outs]
            confs = [float(p.max()) for p in probs]
            e = next(k for k in range(len(probs)) if k == len(probs) - 1 or confs[k] >= thr[k])
            r = {"exit": e, "pred": int(probs[e].argmax()), "confidences": confs[: e + 1] + [None] * (len(probs) - e - 1)}
        else:                                        # static: one head after block 4
            p = m["model"](x, mask)[0].softmax(-1)
            r = {"exit": 3, "pred": int(p.argmax()), "confidences": [None, None, None, float(p.max())]}
        times.append((time.perf_counter() - t0) * 1000.0)
    r["ms"] = statistics.median(times)
    r["cost"] = m["flops"][r["exit"]] if m["adaptive"] else m["flops"][0]
    return r


def simulate(m: dict, split: dict, start: int, n: int, scenario: dict, full_ref: int) -> list[dict]:
    """Run n windows from `start` under a battery scenario. Returns one record per window."""
    X, M = split["X"], split["mask"]
    mode = scenario["mode"]
    battery = float(scenario["level"]) if mode == "fixed" else float(scenario.get("start_battery", 100))
    capacity = scenario.get("capacity_windows", 10) * full_ref        # draining: battery = this much compute
    recs = []
    for w in range(n):
        gi = start + w
        level = scenario["level"] if mode == "fixed" else level_for(battery)
        if mode == "custom":
            thr = [scenario["custom_thr"]] * 3 + [0.0]
        elif m["adaptive"]:
            thr = [float(t) for t in m["thresholds"][level]]
        else:
            thr = None
        rec = {"gi": gi, "battery_in": battery, "level": level, "thr": thr, "dead": battery <= 0.0}
        if not rec["dead"]:
            rec.update(run_window(m, X[gi:gi + 1], M[gi:gi + 1], thr))
            if mode == "draining":
                battery = max(0.0, battery - 100.0 * rec["cost"] / capacity)
        else:
            rec.update({"exit": None, "pred": None, "confidences": [None] * 4, "ms": 0.0, "cost": 0})
        rec["battery_out"] = battery
        recs.append(rec)
    return recs


# ------------------------------------------------------------------ frame renderer

def render_frame(sc: dict, w: int, f: int, cum_before: int) -> Image.Image:
    """Scenario sc, window w (0..n-1) of the clip, frame f (0..T-1) inside it."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    r = sc["records"][w]
    m = sc["model"]
    gi = r["gi"]
    label = int(sc["y"][gi])
    tsec = float(sc["start_frames"][gi]) / FPS

    # titles + battery gauge
    text(d, (40, 28), sc["title"], F_TITLE)
    text(d, (40, 70), f"{sc['source']}  ·  window {gi}  ·  {int(tsec // 60):02d}:{tsec % 60:04.1f}", F_S, MUTED)
    batt = r["battery_in"]
    gx0, gy0, gw, gh = W - 300, 26, 220, 34
    col = GREEN if batt > 50 else (AMBER if batt > 20 else RED)
    box(d, (gx0, gy0, gx0 + gw, gy0 + gh), fill=PANEL, outline=LINE, radius=6)
    if batt > 0:
        d.rounded_rectangle((gx0 + 3, gy0 + 3, gx0 + 3 + (gw - 6) * min(1.0, batt / 100), gy0 + gh - 3), radius=4, fill=col)
    d.rectangle((gx0 + gw + 2, gy0 + 10, gx0 + gw + 8, gy0 + gh - 10), fill=LINE)
    text(d, (gx0 + gw / 2, gy0 + gh / 2), f"battery {batt:.0f}%", F_S, LIGHT, "mm")
    if m["adaptive"]:
        mode_txt = "custom threshold" if sc["mode"] == "custom" else f"policy level {r['level']}%"
    else:
        mode_txt = "static: battery has no effect"
    text(d, (gx0 + gw / 2, gy0 + gh + 18), mode_txt, F_S, MUTED, "mm")

    # camera strip: the frame behind this window frame, with the tracked boxes
    k = w * T + f + sc["frame_offset"]
    cam = sc["camera"]["frames"]
    if k < len(cam):
        img.paste(Image.fromarray(cam[k]), (CX0, CY0))
    frame_no = sc["first_frame"] + k if sc["first_frame"] is not None else None
    geom = sc["camera"]["geom"]
    if geom and frame_no is not None:
        sx, sy = CW / geom["w"], CH / geom["band"]
        for b in sc["boxes"].get(frame_no, []):
            bc = AMBER if b["team"] == "right" else (BLUE if b["team"] == "left" else LIGHT)
            bx0, by0 = CX0 + b["x"] * sx, CY0 + (b["y"] - geom["y0"]) * sy
            if by0 < CY0 or by0 + b["h"] * sy > CY0 + CH:
                continue                             # outside the cropped band
            d.rectangle([bx0, by0, bx0 + b["w"] * sx, by0 + b["h"] * sy], outline=bc, width=1)
    d.rectangle([CX0, CY0, CX0 + CW, CY0 + CH], outline=LINE, width=1)
    cap = f"camera, frame {frame_no}" if frame_no is not None else "uploaded video"
    if sc["boxes"]:
        cap += "   ·   tracked boxes: amber = right team, blue = left team, white = referee"
    tw = d.textlength(cap, font=F_S)
    d.rectangle([CX0 + 1, CY0 + CH - 28, CX0 + tw + 17, CY0 + CH - 1], fill=BG)
    text(d, (CX0 + 9, CY0 + CH - 8), cap, F_S, LIGHT, "ls")

    # pitch + players for this frame (the model input)
    draw_pitch(d)
    xf, mf = sc["X"][gi, f], sc["M"][gi, f]
    for p in range(xf.shape[0]):
        if not bool(mf[p]):
            continue
        px, py = PX0 + float(xf[p, 0]) * PW, PY0 + float(xf[p, 1]) * PH
        pc = AMBER if p < 11 else (BLUE if p < 22 else LIGHT)
        rad = 7 if p < 22 else 5
        d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=pc, outline=BG, width=1)
    ly = PY0 + PH + 10
    if label == 1:
        box(d, (PX0, ly, PX0 + 170, ly + 32), fill=AMBER, outline=AMBER)
        text(d, (PX0 + 85, ly + 16), "ACTION in window", F_S, BG, "mm")
    else:
        box(d, (PX0, ly, PX0 + 170, ly + 32), outline=LINE)
        text(d, (PX0 + 85, ly + 16), "no action", F_S, SOFT, "mm")
    text(d, (PX0 + 186, ly + 16), "bird's-eye view = what\nthe model actually reads", F_S, MUTED, "lm")

    # right panel: trunk
    x0, y0 = 460, 322
    box(d, (x0 - 16, y0 - 12, W - 24, 600), fill=PANEL, outline=PANEL)
    head = "Trunk executed for this window"
    if m["key"] == "int8":
        head += "  ·  INT8 weights"
    elif m["key"] == "pruned" and m["widths"]:
        head += f"  ·  {m['widths'][1]} of 64 channels"
    text(d, (x0, y0), head, F_H, SOFT)
    n_exec = 0 if r["dead"] else (r["exit"] + 1 if m["adaptive"] else 4)
    decided_at = 2 * n_exec + 2
    for kb in range(1, 5):
        by = y0 + 40 + (kb - 1) * 46
        runs = kb <= n_exec and f >= 2 * kb
        pending = kb <= n_exec and f < 2 * kb
        fill = GREEN if runs else (PANEL if pending else None)
        box(d, (x0, by, x0 + 130, by + 38), fill=fill, outline=GREEN if (runs or pending) else LINE)
        text(d, (x0 + 65, by + 19), f"Block {kb}", F_B, LIGHT if runs else (SOFT if pending else MUTED), "mm")
        if kb > n_exec:
            text(d, (x0 + 150, by + 19), "not executed", F_S, MUTED, "lm")
            continue
        if not runs:
            continue
        if m["adaptive"]:
            c = r["confidences"][kb - 1]
            taken = (kb - 1) == r["exit"]
            if taken and kb == 4:
                text(d, (x0 + 150, by + 19), f"exit 4: conf {c:.3f}, final exit  →  STOP", F_S, AMBER, "lm")
            elif taken:
                text(d, (x0 + 150, by + 19), f"exit {kb}: conf {c:.3f} ≥ {r['thr'][kb - 1]:.2f}  →  STOP", F_S, AMBER, "lm")
            else:
                text(d, (x0 + 150, by + 19), f"exit {kb}: conf {c:.3f} < {r['thr'][kb - 1]:.2f}  →  continue", F_S, SOFT, "lm")
        elif kb == 4:
            text(d, (x0 + 150, by + 19), f"single head: conf {r['confidences'][3]:.3f}", F_S, SOFT, "lm")

    # decision
    dy = y0 + 40 + 4 * 46 + 6
    if r["dead"]:
        box(d, (x0, dy, W - 40, dy + 40), outline=RED)
        text(d, (x0 + 14, dy + 20), "battery empty: no inference", F_H, RED, "lm")
    elif f >= decided_at:
        ok = r["pred"] == label
        dc = GREEN if ok else RED
        box(d, (x0, dy, W - 40, dy + 40), outline=dc)
        text(d, (x0 + 14, dy + 20), f"prediction: {'ACTION' if r['pred'] else 'no action'}", F_H, dc, "lm")
        text(d, (W - 54, dy + 20), "correct" if ok else "wrong", F_S, dc, "rm")
        share = r["cost"] / sc["full_ref"]
        text(d, (W - 130, dy + 20), f"{r['cost']:,} FLOPs ({100 * share:.0f}%)  ·  {r['ms']:.2f} ms", F_S, SOFT, "rm")
    else:
        box(d, (x0, dy, W - 40, dy + 40), outline=LINE)
        text(d, (x0 + 14, dy + 20), "running ...", F_H, MUTED, "lm")

    # bottom strip: cumulative compute + timeline
    n = len(sc["records"])
    sy0 = 608
    cum = cum_before + (r["cost"] if (not r["dead"] and f >= decided_at) else 0)
    ref = n * sc["full_ref"]
    text(d, (40, sy0), f"compute so far: {cum / 1e6:.2f} M FLOPs", F_B, LIGHT)
    text(d, (W - 40, sy0), f"full CNN on all {n} windows: {ref / 1e6:.2f} M FLOPs", F_S, MUTED, "ra")
    bx0, bx1, byy = 40, W - 40, sy0 + 32
    d.rectangle([bx0, byy, bx1, byy + 16], fill=PANEL, outline=LINE)
    fillw = int((bx1 - bx0) * min(1.0, cum / ref))
    if fillw > 0:
        d.rectangle([bx0, byy, bx0 + fillw, byy + 16], fill=GREEN if m["adaptive"] else AMBER)
    ty = byy + 26
    sq = (bx1 - bx0 - (n - 1) * 6) / n
    for j in range(n):
        qx = bx0 + j * (sq + 6)
        rj = sc["records"][j]
        done = j < w or (j == w and (rj["dead"] or f >= decided_at))
        if done and rj["dead"]:
            d.rectangle([qx, ty, qx + sq, ty + 12], fill=RED)
        elif done:
            shade = EXIT_SHADES[rj["exit"]] if m["adaptive"] else AMBER
            d.rectangle([qx, ty, qx + sq, ty + 12], fill=shade)
        else:
            d.rectangle([qx, ty, qx + sq, ty + 12], outline=LINE)
        if int(sc["y"][rj["gi"]]) == 1:
            d.polygon([(qx + sq / 2 - 5, ty + 23), (qx + sq / 2 + 5, ty + 23), (qx + sq / 2, ty + 16)], fill=AMBER)
    if m["adaptive"]:
        legend = "timeline: dark green = exit 1, lighter = exit 2 or 3, amber = exit 4, red = battery empty   ·   marker = action"
    else:
        legend = "timeline: every window runs all four blocks, red = battery empty   ·   marker below = action window"
    text(d, (40, ty + 26), legend, F_S, MUTED)
    return img


def cum_costs(records: list[dict]) -> list[int]:
    out, c = [], 0
    for r in records:
        out.append(c)
        c += r["cost"]
    return out


def write_video(sc: dict, progress) -> bytes:
    n = len(sc["records"])
    before = cum_costs(sc["records"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
        if not vw.isOpened():                        # avc1 (H.264) plays in browsers; mp4v is the fallback
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for w in range(n):
            for f in range(T):
                vw.write(cv2.cvtColor(np.asarray(render_frame(sc, w, f, before[w])), cv2.COLOR_RGB2BGR))
            progress.progress((w + 1) / n, text=f"rendering window {w + 1}/{n}")
        vw.release()
        return path.read_bytes()


# ------------------------------------------------------------------ app

st.set_page_config(page_title="Adaptive inference demo · BDE group 21", layout="wide")
st.title("Sports action recognition at the edge")
st.caption("Big Data at the Edge, group 21  ·  SoccerTrack v2  ·  full vs pruned vs early-exit vs INT8, under different battery levels")
st.markdown(
    "**About this demo.** This page is a visualisation only. It does not train anything and it does not change the "
    "models: it loads the trained checkpoints from `outputs/`, runs them on the player positions of one clip, and shows "
    "what each model predicts and how much compute it spends under the battery scenario you pick in the sidebar. "
    "Predictions and latencies are computed live from those checkpoints; the exit thresholds and FLOP counts come from "
    "the evaluation in the notebooks. Nothing here is a new result, it only makes the behaviour visible.\n\n"
    "**Everything you see is drawn from the data, not generated.** The clip is cut into fixed-length windows of "
    "tracking data, and every frame of the video is composed from three sources. The camera strip at the top is the "
    "real broadcast frame decoded from the clip with OpenCV, cropped to the band where the players are. The coloured "
    "boxes on it are the SoccerTrack tracking annotations for that exact frame, scaled from the native camera "
    "resolution (amber = right team, blue = left team, white = referee). The bird's-eye pitch is the same tracking data "
    "as normalised pitch coordinates, which is exactly the tensor the model reads; the camera image is never fed to the "
    "model. Everything else in the frame comes from the simulation of that window: the battery gauge and the scenario "
    "mode, and then the model's own output: the trunk block by block with the confidence at each early exit and where "
    "the policy decided to stop, the prediction against the ground-truth label, and the running compute compared with "
    "running the full CNN on every window.\n\n"
    "**How the video is created.** The app runs the selected model window by window with the thresholds for the "
    "current battery level and records exit, confidence, prediction, FLOPs and latency per window. It then draws one "
    "image per frame with Pillow, combining the decoded camera frame, the tracking boxes and positions, and those "
    "records. You can scrub through these frames directly with the slider, or press **Render video** to encode all of "
    "them into an mp4 with OpenCV at the clip's native frame rate and download it."
)

loaded = load_models()
full_ref = loaded["full_ref"]

with st.sidebar:
    st.header("1 · Clip")
    source = st.radio("Video", ["SoccerTrack clip", "Upload a video"], label_visibility="collapsed")
    video_path, video_name, boxes = None, None, {}
    if source == "SoccerTrack clip":
        clips = sorted(FRAMES_DIR.glob("*.mp4"), key=lambda p: parse_clip_name(p.name) or ("", 0, 0))
        if clips:
            by_name = {p.name: p for p in clips}
            # The choice lives in session_state under a key (the file name), so it survives reruns and
            # clips being added to the folder; it only falls back to the default if the file is gone.
            # clip_saved is only written by the user (on_change), so a reset caused by new options is undone.
            saved = st.session_state.get("clip_saved")
            if saved not in by_name:
                saved = DEFAULT_VIDEO.name if DEFAULT_VIDEO.name in by_name else clips[0].name
            st.session_state["clip_choice"] = saved
            chosen = by_name[st.selectbox("Clip", list(by_name), key="clip_choice", format_func=lambda n: clip_label(by_name[n]),
                                          on_change=lambda: st.session_state.update(clip_saved=st.session_state["clip_choice"]),
                                          help=f"every .mp4 in {FRAMES_DIR.relative_to(REPO)} "
                                               "(fetch more with scripts/fetch_demo_frames.py --first <frame>)")]
            video_path, video_name = str(chosen), chosen.name
            bpath = chosen.with_name(chosen.stem + "_boxes.json")
            if bpath.exists():
                boxes = {int(k): v for k, v in json.loads(bpath.read_text()).items()}
        else:
            st.error(f"no clip in {FRAMES_DIR.relative_to(REPO)}; run scripts/fetch_demo_frames.py or upload a video")
    else:
        up = st.file_uploader("Video (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"])
        bup = st.file_uploader("Tracked boxes JSON (optional)", type=["json"],
                               help="the <clip>_boxes.json written by scripts/fetch_demo_frames.py")
        if up is not None:
            data = up.getvalue()
            tmpdir = Path(tempfile.gettempdir()) / "bde_streamlit"
            tmpdir.mkdir(exist_ok=True)
            p = tmpdir / f"{file_digest(data)}{Path(up.name).suffix}"
            if not p.exists():
                p.write_bytes(data)
            video_path, video_name = str(p), up.name
        if bup is not None:
            boxes = {int(k): v for k, v in json.loads(bup.getvalue()).items()}

    st.header("2 · Tracking data")
    all_splits = sorted((REPO / "data" / "processed").glob("*.pt"))
    lengths = {str(p.relative_to(REPO)): window_length(str(p), p.stat().st_mtime) for p in all_splits}
    splits = [s for s, t in lengths.items() if t == T]
    skipped = [s for s, t in lengths.items() if t != T]
    if not splits:
        st.error(f"no split in data/processed has {T}-frame windows; the checkpoints need data/processed/demo_117093_half2.pt")
        st.stop()
    parsed_clip = parse_clip_name(video_name) if video_name else None
    covering = [s for s in splits
                if parsed_clip and parsed_clip[:2] in split_halves(str(REPO / s), (REPO / s).stat().st_mtime)]
    if covering:                                     # a SoccerTrack clip: use the split of its match and half
        split_rel = covering[0]
        st.caption(f"Player positions for this clip: `{split_rel}` (picked automatically from its match and half; "
                   f"{len(splits)} compatible files: {', '.join(Path(s).stem for s in splits)}). "
                   "The model itself is chosen under 3 · Scenario.")
    else:
        if parsed_clip:
            st.warning(f"No {T}-frame split covers match {parsed_clip[0]}, half {parsed_clip[1]}; pick windows from another split.")
        default_split = str(DEFAULT_SPLIT.relative_to(REPO))
        split_rel = st.selectbox("Split (model input)", splits, index=splits.index(default_split) if default_split in splits else 0,
                                 help=f"only splits with {T}-frame windows are listed: the committed checkpoints were trained on those")
    if parsed_clip and parsed_clip[0] in SPLIT_OF_MATCH:
        role = SPLIT_OF_MATCH[parsed_clip[0]]
        st.caption(f"Match {parsed_clip[0]} is a **{role}** match." + (
            " The models were trained on it, so it is not a fair accuracy test "
            "(in practice bal_acc is about 0.55 here too, the same as on unseen matches)." if role == "training" else ""))
    if skipped:
        st.caption(f"{len(skipped)} local split(s) hidden: they have other window lengths "
                   f"({', '.join(sorted({f'T={lengths[s]}' for s in skipped}))}) and do not fit the checkpoints.")

    st.header("3 · Scenario")
    model_name = st.radio("Model", list(MODELS), help="\n".join(f"**{k}**: {v['blurb']}" for k, v in MODELS.items()))
    m = loaded["models"][model_name]
    mode_label = st.radio("Battery", ["Fixed level", "Draining battery", "Custom threshold"],
                          help="Fixed: the controller uses the thresholds tuned for that level. Draining: the battery "
                               "drops with every FLOP and the controller switches at 50% and 20%. Custom: one threshold for exits 1-3.")
    scenario = {"mode": {"Fixed level": "fixed", "Draining battery": "draining", "Custom threshold": "custom"}[mode_label], "level": "100"}
    if scenario["mode"] == "fixed":
        scenario["level"] = st.select_slider("Battery level", options=["20", "50", "100"], value="100")
    elif scenario["mode"] == "draining":
        scenario["start_battery"] = st.slider("Start battery (%)", 5, 100, 60)
        scenario["capacity_windows"] = st.slider("Full charge = full CNN on this many windows", 2, 60, 12,
                                                 help="smaller = the battery empties sooner")
    else:
        scenario["custom_thr"] = st.slider("Threshold for exits 1-3", 0.50, 1.00, 0.55, 0.005)
    if m["adaptive"] and scenario["mode"] != "custom":
        st.caption("Thresholds: " + "  ·  ".join(f"{lv}%: {m['thresholds'][lv]}" for lv in LEVELS))
    if not m["adaptive"]:
        st.caption("Static model: the policy does not apply, it always runs all four blocks. "
                   "With a draining battery you can see how fast it empties.")

if video_path is None:
    st.info("Choose the bundled clip or upload a video in the sidebar.")
    st.stop()

split = load_tracking(str(REPO / split_rel))
if split["X"].shape[1] != T:
    st.error(f"{split_rel} has {split['X'].shape[1]}-frame windows; the checkpoints expect {T}. "
             f"Use data/processed/demo_117093_half2.pt (the local train/val/test files were rebuilt with another window length).")
    st.stop()

# Some matches (e.g. 117092) ship image boxes that are all the same size and do not line up with the
# video (generated, not detected). They still locate the players roughly, so they position the crop,
# but they are not drawn.
crop_boxes = boxes
if boxes and statistics.pstdev([b["w"] for v in boxes.values() for b in v]) < 2.0:
    boxes = {}
    with st.sidebar:
        st.caption("Tracked boxes hidden for this clip: this match's image-space boxes do not line up with its "
                   "video. The bird's-eye pitch (the model input) is unaffected.")
geom = None
if parse_clip_name(video_name):                     # SoccerTrack clip: crop the pitch band, boxes line up
    _cap = cv2.VideoCapture(video_path)
    _fw, _fh = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap.release()
    if _fw and _fh:
        _match, _half, _ = parse_clip_name(video_name)
        geom = crop_geometry(native_size(_match, _half, _fw, _fh), crop_boxes)
camera = load_camera(video_path, video_name + str(Path(video_path).stat().st_size), MAX_WINDOWS * T, geom)
if not camera["frames"]:
    st.error("could not decode any frame from this video")
    st.stop()

# align the clip with tracking windows
parsed = parse_clip_name(video_name)
first_frame, frame_offset, start = None, 0, None
if parsed:
    match, half, first_frame = parsed
    start, offset = find_window(split, match, half, first_frame)
    if start is not None and offset:                 # clip starts mid-window: skip to the next window boundary
        start, frame_offset = start + 1, T - offset
if start is None:
    with st.sidebar:
        st.warning("This video is not linked to tracking windows (name is not <match>_<half>_<first_frame>_<n>). "
                   "Pick the windows to run; the predictions are for those windows, not for the video pixels.")
        start = st.number_input("First tracking window", 0, len(split["y"]) - 1, 0)
    first_frame = None
n_win = min((len(camera["frames"]) - frame_offset) // T, len(split["y"]) - start, MAX_WINDOWS)
if n_win < 1:
    st.error("the clip is shorter than one 12-frame window")
    st.stop()

meta = split["meta"]
half_txt = "1st" if int(meta["half"][start]) == 1 else "2nd"
records = simulate(m, split, start, n_win, scenario, full_ref)
if m["adaptive"]:
    scen_txt = {"fixed": f"battery {scenario['level']}%", "draining": f"draining battery from {scenario.get('start_battery')}%",
                "custom": f"threshold {scenario.get('custom_thr', 0):.3f}"}[scenario["mode"]]
else:
    scen_txt = "always full depth" if scenario["mode"] != "draining" else f"draining battery from {scenario['start_battery']}%"
sc = dict(
    model=m, records=records, mode=scenario["mode"], title=f"{model_name}  ·  {scen_txt}",
    source=f"SoccerTrack v2, match {meta['match_id'][start]}, {half_txt} half",
    X=split["X"], M=split["mask"], y=split["y"], start_frames=meta["start_frame"],
    camera=camera, boxes=boxes, first_frame=first_frame, frame_offset=frame_offset, full_ref=full_ref,
)

# ---- summary metrics
live = [r for r in records if not r["dead"]]
correct = sum(int(r["pred"] == int(split["y"][r["gi"]])) for r in live)
mean_cost = sum(r["cost"] for r in live) / max(1, len(live))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Windows run", f"{len(live)}/{n_win}", help="windows after the battery ran out are not processed")
c2.metric("Mean compute / window", f"{mean_cost / 1e3:,.0f} kFLOPs", f"{100 * mean_cost / full_ref:.0f}% of full CNN", delta_color="off")
c3.metric("Median latency / window", f"{statistics.median([r['ms'] for r in live]) if live else 0:.3f} ms",
          help="single CPU thread, median of 5 runs per window" + ("; INT8 graph always computes every exit" if m["key"] == "int8" else ""))
c4.metric("Correct on this clip", f"{correct}/{len(live)}")
c5.metric("Model size", f"{m['size_mb']:.3f} MB")

vid_key = (model_name, json.dumps(scenario, sort_keys=True), video_name, split_rel, start, n_win)
left, right = st.columns([3, 1])
with right:
    st.markdown("**Render the clip**")
    st.caption(f"{n_win} windows × {T} frames = {n_win * T / FPS:.1f} s at {FPS} fps, same layout as the demo clips.")
    if st.button("Render video", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="rendering ...")
        videos = st.session_state.setdefault("videos", {})     # every rendered scenario, newest last
        videos[vid_key] = write_video(sc, bar)
        while len(videos) > 8:
            videos.pop(next(iter(videos)))
        bar.empty()
    have = st.session_state.get("videos", {}).get(vid_key)
    if have:
        st.download_button("Download mp4", have, file_name=f"demo_{m['key']}_{scenario['mode']}.mp4",
                           mime="video/mp4", use_container_width=True)
with left:
    have = st.session_state.get("videos", {}).get(vid_key)
    if have:
        st.video(have)
    else:
        k = st.slider("Frame in clip (scrub)", 0, n_win * T - 1, min(8 * T + T - 1, n_win * T - 1))
        w, f = divmod(k, T)
        st.image(render_frame(sc, w, f, cum_costs(records)[w]), use_container_width=True)
        st.caption("Scrub through the clip, or press **Render video** for the full animation.")

with st.expander("Per-window trace"):
    rows = []
    for r in records:
        rows.append({
            "window": r["gi"], "label": "action" if int(split["y"][r["gi"]]) else "none",
            "battery in %": round(r["battery_in"], 1), "level": r["level"] if m["adaptive"] else None,
            "exit": None if r["dead"] else r["exit"] + 1,
            **{f"conf {e + 1}": (None if r["confidences"][e] is None else round(r["confidences"][e], 3)) for e in range(4)},
            "pred": "battery empty" if r["dead"] else ("action" if r["pred"] else "none"),
            "FLOPs": r["cost"], "ms": round(r["ms"], 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
