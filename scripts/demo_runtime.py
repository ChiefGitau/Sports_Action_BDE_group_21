"""Runtime demo: watch the early-exit model decide, one window at a time.

Two demos, both on real windows and the committed M3 checkpoint:

  trace    stream a handful of windows through the trunk block by block. After every exit
           head the confidence is checked against the threshold for the current budget; the
           trunk really stops when it is high enough. Prints one row per window.

  battery  simulate an edge box over a stretch of a match. The battery drains in proportion
           to the FLOPs spent; when it crosses 50% and 20% the controller switches to the
           thresholds tuned for that budget. Prints one row per battery segment.

Examples:
    python -m scripts.demo_runtime trace   --split data/processed/val.pt --windows 10 --start 4200
    python -m scripts.demo_runtime battery --split data/processed/val.pt --capacity-fraction 0.35

The policy and thresholds are the ones used in M5/M7 (outputs/earlyexit/thresholds.json);
nothing is re-tuned here. Any .pt in the Core data contract works as --split, so a single
freshly built match half is enough for a live demo.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor

from core import compute_metrics, load_flops, load_split, load_thresholds
from core.policy import budget_thresholds
from models.cnn import ExitCNN, featurize
from models.registry import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
FPS = 25


# --------------------------------------------------------------------------------------
# One window, truly incremental
# --------------------------------------------------------------------------------------

@torch.no_grad()
def stream_window(model: ExitCNN, x: Tensor, mask: Tensor, thresholds: list[float]) -> dict:
    """Run one window [1, T, P, 2] through the trunk and stop at the first confident exit.

    Unlike core.policy.apply_threshold_policy (which chooses an exit from logits that were
    all computed), this is what the deployed device does: later blocks are never executed
    once an exit is taken.
    """
    t0 = time.perf_counter()
    h = model.embed(featurize(x, mask))
    confidences: list[float | None] = [None] * model.num_exits
    head = 0
    for i, block in enumerate(model.blocks, start=1):
        h = block(h)
        if head < model.num_exits and i == model.exit_after[head]:
            logits = model.heads[head](h)
            probs = logits.softmax(dim=-1)
            conf = float(probs.max())
            confidences[head] = conf
            last = head == model.num_exits - 1
            if last or conf >= thresholds[head]:
                return {
                    "exit": head,
                    "pred": int(probs.argmax()),
                    "p_action": float(probs[0, 1]),
                    "confidences": confidences,
                    "ms": (time.perf_counter() - t0) * 1000.0,
                }
            head += 1
    raise RuntimeError("unreachable: the last exit always answers")


# --------------------------------------------------------------------------------------
# Demo 1: trace a few windows
# --------------------------------------------------------------------------------------

def fmt_conf(c: float | None, threshold: float, taken: bool) -> str:
    if c is None:
        return "   --   "
    mark = "*" if taken else (" " if c < threshold else " ")
    return f"{c:.3f}{mark}  "


def demo_trace(model: ExitCNN, split: dict, flops: list[int], thresholds: list[float], start: int, n: int, budget: str) -> None:
    X, M, y = split["X"], split["mask"], split["y"]
    frames = split.get("meta", {}).get("start_frame")
    full = flops[-1]
    n = min(n, len(y) - start)

    print(f"\nTRACE  budget={budget}%  thresholds={thresholds}  windows {start}..{start + n - 1}")
    print(f"  (a window exits at the first head whose max-softmax >= threshold; * marks the exit taken)\n")
    head = f"{'win':>6} {'t(s)':>7} {'label':>6} | {'exit1':>8} {'exit2':>8} {'exit3':>8} {'exit4':>8} | {'exit':>4} {'FLOPs':>9} {'%full':>5} {'pred':>6} {'ok':>3} {'ms':>6}"
    print(head)
    print("-" * len(head))
    used, correct = [], 0
    for i in range(start, start + n):
        r = stream_window(model, X[i : i + 1], M[i : i + 1], thresholds)
        t = f"{int(frames[i]) / FPS:7.1f}" if frames is not None else f"{'':>7}"
        label = "action" if int(y[i]) == 1 else "none"
        pred = "action" if r["pred"] == 1 else "none"
        ok = "yes" if r["pred"] == int(y[i]) else "no"
        correct += ok == "yes"
        used.append(flops[r["exit"]])
        confs = " ".join(fmt_conf(c, thresholds[k], k == r["exit"]) for k, c in enumerate(r["confidences"]))
        print(f"{i:>6} {t} {label:>6} | {confs}| {r['exit'] + 1:>4} {flops[r['exit']]:>9,} {100 * flops[r['exit']] / full:5.1f} {pred:>6} {ok:>3} {r['ms']:6.3f}")
    mean = sum(used) / len(used)
    print("-" * len(head))
    print(f"{'mean':>21} | {'':>36}| {'':>4} {mean:>9,.0f} {100 * mean / full:5.1f} {'':>6} {correct}/{n}\n")


# --------------------------------------------------------------------------------------
# Demo 2: battery draining over a stretch of the match
# --------------------------------------------------------------------------------------

def level_for(battery: float) -> str:
    if battery > 50:
        return "100"
    if battery > 20:
        return "50"
    return "20"


def demo_battery(model: ExitCNN, split: dict, flops: list[int], thr_by_budget: dict[str, list[float]], start: int, n: int, capacity_fraction: float) -> None:
    X, M, y = split["X"], split["mask"], split["y"]
    n = min(n, len(y) - start) if n > 0 else len(y) - start
    full = flops[-1]
    # capacity: running the full model on every window would drain this fraction of the
    # stretch; smaller fractions make the battery cross 50% and 20% sooner.
    capacity = capacity_fraction * n * full
    battery = 100.0

    print(f"\nBATTERY  windows {start}..{start + n - 1}  capacity = full model on {capacity_fraction:.0%} of them")
    print("  level 100% uses thresholds", thr_by_budget["100"], "| 50%:", thr_by_budget["50"], "| 20%:", thr_by_budget["20"])
    print("  (the 20% level has no feasible threshold set, so the controller falls back to exit 1 for every window)\n")

    segments: list[dict] = []
    cur = None
    for i in range(start, start + n):
        level = level_for(battery)
        if cur is None or cur["level"] != level:
            cur = {"level": level, "from": i, "battery_in": battery, "exits": [0] * model.num_exits, "flops": 0, "logits": [], "y": []}
            segments.append(cur)
        r = stream_window(model, X[i : i + 1], M[i : i + 1], thr_by_budget[level])
        cost = flops[r["exit"]]
        battery = max(0.0, battery - 100.0 * cost / capacity)
        cur["exits"][r["exit"]] += 1
        cur["flops"] += cost
        cur["to"] = i
        cur["battery_out"] = battery
        cur["logits"].append([1.0 - r["p_action"], r["p_action"]])
        cur["y"].append(int(y[i]))
        if battery <= 0.0:
            break

    head = f"{'level':>5} {'windows':>13} {'battery':>12} {'n':>5} {'mean FLOPs':>11} {'%full':>5} {'exit1':>6} {'exit2':>6} {'exit3':>6} {'exit4':>6} {'bal_acc':>7} {'f1':>6}"
    print(head)
    print("-" * len(head))
    for s in segments:
        cnt = sum(s["exits"])
        mean = s["flops"] / cnt
        m = compute_metrics(torch.tensor(s["logits"]), torch.tensor(s["y"]), 2)
        print(f"{s['level'] + '%':>5} {s['from']:>6}-{s['to']:<6} {s['battery_in']:5.1f}->{s['battery_out']:5.1f} {cnt:>5} {mean:>11,.0f} {100 * mean / full:5.1f} "
              f"{s['exits'][0]:>6} {s['exits'][1]:>6} {s['exits'][2]:>6} {s['exits'][3]:>6} {m['bal_acc']:7.3f} {m['f1']:6.3f}")
    total_windows = sum(sum(s["exits"]) for s in segments)
    total_flops = sum(s["flops"] for s in segments)
    print("-" * len(head))
    print(f"{'all':>5} {'':>13} {'':>12} {total_windows:>5} {total_flops / total_windows:>11,.0f} {100 * total_flops / total_windows / full:5.1f}"
          f"   vs {full:,} ({100.0:.1f}) for always running to exit 4\n")


# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["trace", "battery"])
    p.add_argument("--split", required=True, help="a .pt file in the Core data contract (e.g. data/processed/val.pt)")
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/earlyexit/model.pt"))
    p.add_argument("--thresholds", default=str(ROOT / "outputs/earlyexit/thresholds.json"))
    p.add_argument("--flops", default=str(ROOT / "outputs/earlyexit/flops.json"))
    p.add_argument("--start", type=int, default=0, help="first window index")
    p.add_argument("--windows", type=int, default=10, help="trace: how many windows; battery: 0 = until the end of the split")
    p.add_argument("--budget", default="100", choices=["100", "50", "20"], help="trace: battery level to use")
    p.add_argument("--capacity-fraction", type=float, default=0.35, help="battery: see demo_battery")
    args = p.parse_args()

    torch.set_num_threads(1)                      # the edge scenario: one window, one core
    model, info = load_checkpoint(args.checkpoint)
    if not isinstance(model, ExitCNN) or model.num_exits < 2:
        raise SystemExit("demo needs the early-exit checkpoint (outputs/earlyexit/model.pt)")
    split = load_split(args.split)
    flops = [int(f) for f in load_flops(args.flops)["flops_per_exit"]]
    thr_by_budget = load_thresholds(args.thresholds)

    T, P = split["X"].shape[1], split["X"].shape[2]
    for _ in range(5):                            # warm-up so the first timed window is not an outlier
        stream_window(model, split["X"][:1], split["mask"][:1], [0.0] * model.num_exits)
    print(f"model: {info['kind']} {info['variant']}  exits after blocks {model.exit_after}  FLOPs per exit {flops}")
    print(f"split: {Path(args.split).name}  {len(split['y']):,} windows of {T} frames x {P} slots, "
          f"{100 * float(split['y'].float().mean()):.1f}% action")

    if args.mode == "trace":
        demo_trace(model, split, flops, budget_thresholds(thr_by_budget, args.budget), args.start, args.windows, args.budget)
    else:
        demo_battery(model, split, flops, thr_by_budget, args.start, args.windows, args.capacity_fraction)


if __name__ == "__main__":
    main()
