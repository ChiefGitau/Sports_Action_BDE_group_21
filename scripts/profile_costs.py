from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import models

from core.config import load_config
from core.data import load_split
from core.flops import profile
from core.model import DummyExitModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the computational cost of the early-exit model."
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/processed/val.pt"),
        help="Processed dataset used to obtain one real input window.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint/state_dict.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flops.json"),
        help="JSON output file.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("profiling_report.txt"),
        help="Human-readable profiling report.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of CPU latency warm-up iterations.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of measured CPU latency iterations.",
    )

    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Checkpoint must contain a PyTorch state_dict or a dictionary "
            "containing 'model_state_dict'/'state_dict'."
        )

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def latency_per_exit(
    model: DummyExitModel,
    x: torch.Tensor,
    mask: torch.Tensor,
    warmup: int,
    repeats: int,
) -> list[dict[str, float]]:
    """
    Measure CPU latency for batch size 1.

    Each exit is measured through forward_until(), so the latency represents
    the cumulative computation required to reach that exit.
    """

    was_training = model.training
    model.eval()

    x = x[:1].cpu()
    mask = mask[:1].cpu()
    model = model.cpu()

    results: list[dict[str, float]] = []

    with torch.no_grad():
        for exit_idx in range(model.num_exits):

            for _ in range(warmup):
                model.forward_until(x, mask, exit_idx)

            timings_ms: list[float] = []

            for _ in range(repeats):
                start = time.perf_counter()
                model.forward_until(x, mask, exit_idx)
                end = time.perf_counter()

                timings_ms.append((end - start) * 1000.0)

            results.append(
                {
                    "exit": exit_idx + 1,
                    "mean_ms": statistics.mean(timings_ms),
                    "median_ms": statistics.median(timings_ms),
                    "min_ms": min(timings_ms),
                    "max_ms": max(timings_ms),
                }
            )

    model.train(was_training)

    return results


def build_report(results: dict) -> str:
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("M4 COST PROFILING REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Model: {results['model']}")
    lines.append(f"Device: {results['device']}")
    lines.append(f"Input shape: {results['input_shape']}")
    lines.append(f"Mask shape: {results['mask_shape']}")
    lines.append(f"FLOPs tool: {results['flops_tool']}")
    lines.append("")

    lines.append("MODEL STORAGE")
    lines.append("-" * 70)
    lines.append(f"Parameters: {results['params']:,}")
    lines.append(f"Serialized state_dict size: {results['size_mb']:.4f} MB")
    lines.append("")

    lines.append("COMPUTE PER EXIT")
    lines.append("-" * 70)

    flops = results["flops_per_exit"]

    for index, value in enumerate(flops, start=1):
        lines.append(
            f"Exit {index}: "
            f"{value:,} FLOPs "
            f"({value / 1_000_000:.4f} MFLOPs)"
        )

    lines.append("")
    lines.append("CPU LATENCY — BATCH SIZE 1")
    lines.append("-" * 70)

    for item in results["cpu_latency_batch1"]:
        lines.append(
            f"Exit {item['exit']}: "
            f"mean={item['mean_ms']:.4f} ms, "
            f"median={item['median_ms']:.4f} ms, "
            f"min={item['min_ms']:.4f} ms, "
            f"max={item['max_ms']:.4f} ms"
        )

    lines.append("")
    lines.append("NOTES")
    lines.append("-" * 70)
    lines.append(
        "FLOPs and latency are cumulative: Exit N includes the computation "
        "required to reach Exit N."
    )
    lines.append(
        "Latency is measured on CPU with batch size 1 to approximate the "
        "streaming inference scenario."
    )
    lines.append(
        "Final weight-sharing verification will be performed against the "
        "trained M3 early-exit checkpoint."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    cfg = load_config()

    print("=" * 70)
    print("M4 COST PROFILER")
    print("=" * 70)

    if not args.split.exists():
        raise FileNotFoundError(
            f"Processed split not found: {args.split}\n"
            "Expected a processed train/validation/test .pt file."
        )

    print(f"\nLoading profiling sample from: {args.split}")

    data = load_split(
        args.split,
        num_classes=cfg.data.num_classes,
    )

    x = data["X"][:1]
    mask = data["mask"][:1]

    print("sample X:", tuple(x.shape))
    print("sample mask:", tuple(mask.shape))

    # Without a checkpoint, keep the lightweight dummy model so M4 can
    # still be tested independently of the trained M3 model.
    if args.checkpoint is None:
        model = DummyExitModel(
            max_people=cfg.data.max_people,
            num_classes=cfg.data.num_classes,
            num_exits=cfg.model.num_exits,
            hidden=cfg.model.hidden,
        )
        print("No checkpoint supplied: using DummyExitModel.")

    # A real M3 checkpoint stores its architecture together with its
    # state_dict. Reconstruct that architecture using the shared loader.
    else:
        if not args.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}"
            )

        model, checkpoint_meta = models.load_checkpoint(args.checkpoint)

        print(f"Loaded checkpoint: {args.checkpoint}")
        print(
            "Loaded model:",
            type(model).__name__,
            f"variant={checkpoint_meta.get('variant', 'unknown')}",
        )

    model.eval()

    print("\nProfiling FLOPs, parameters and serialized size...")

    base_profile = profile(model, x, mask)

    print("Measuring CPU latency at batch size 1...")

    latency = latency_per_exit(
        model=model,
        x=x,
        mask=mask,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    results = {
        "model": type(model).__name__,
        "device": "cpu",
        "input_shape": list(x.shape),
        "mask_shape": list(mask.shape),
        "flops_tool": base_profile["tool"],
        "flops_per_exit": base_profile["flops_per_exit"],
        "params": base_profile["params"],
        "size_mb": base_profile["size_mb"],
        "cpu_latency_batch1": latency,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    report = build_report(results)

    with args.report.open("w", encoding="utf-8") as file:
        file.write(report)

    print()
    print(report)

    print("=" * 70)
    print("PROFILING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"JSON:   {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()