from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.artifacts import load_flops, load_logits, save_thresholds
from core.metrics import compute_metrics
from core.policy import (
    apply_threshold_policy,
    exit_histogram,
    mean_flops,
)


DEFAULT_BUDGETS = {
    "100": 1.00,
    "50": 0.50,
    "20": 0.20,
}


def make_threshold_grid(step: float) -> list[float]:
    if not 0.0 < step <= 1.0:
        raise ValueError("grid step must be in (0, 1]")

    values = []
    x = 0.0

    while x < 1.0:
        values.append(round(x, 10))
        x += step

    if values[-1] != 1.0:
        values.append(1.0)

    return values


def evaluate_policy(
    logits: torch.Tensor,
    y: torch.Tensor,
    thresholds: list[float],
    flops_per_exit: list[int],
    *,
    method: str,
    temperature: float,
) -> dict:
    predictions, exit_ids, confidence = apply_threshold_policy(
        logits,
        thresholds,
        method=method,
        temperature=temperature,
    )

    sample_ids = torch.arange(
        logits.shape[1],
        device=logits.device,
    )

    selected_logits = logits[
        exit_ids,
        sample_ids,
        :,
    ]

    metrics = compute_metrics(
        selected_logits.cpu(),
        y.cpu(),
    )

    avg_flops = mean_flops(
        exit_ids,
        flops_per_exit,
    )

    return {
        "thresholds": thresholds,
        "mean_flops": float(avg_flops),
        "acc": float(metrics["acc"]),
        "bal_acc": float(metrics["bal_acc"]),
        "f1": float(metrics["f1"]),
        "exit_counts": exit_histogram(
            exit_ids,
            logits.shape[0],
        ),
    }


def choose_for_budget(
    evaluations: list[dict],
    *,
    target_flops: float,
) -> tuple[dict, bool]:
    """
    Pick the best validation operating point that satisfies
    the requested compute budget.

    Primary objective:
        highest balanced accuracy

    Tie breakers:
        F1
        accuracy
        lower FLOPs

    If the requested target is impossible, return the
    lowest-compute operating point instead.
    """
    feasible = [
        result
        for result in evaluations
        if result["mean_flops"] <= target_flops + 1e-9
    ]

    if feasible:
        best = max(
            feasible,
            key=lambda r: (
                r["bal_acc"],
                r["f1"],
                r["acc"],
                -r["mean_flops"],
            ),
        )
        return best, True

    best = min(
        evaluations,
        key=lambda r: (
            r["mean_flops"],
            -r["bal_acc"],
            -r["f1"],
            -r["acc"],
        ),
    )

    return best, False


def tune_thresholds(
    logits: torch.Tensor,
    y: torch.Tensor,
    flops_per_exit: list[int],
    *,
    method: str = "max_softmax",
    temperature: float = 1.0,
    grid_step: float = 0.02,
) -> tuple[dict[str, list[float]], str]:
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [E, N, C], "
            f"got {tuple(logits.shape)}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must have shape [N], got {tuple(y.shape)}"
        )

    num_exits, num_samples, _ = logits.shape

    if y.shape[0] != num_samples:
        raise ValueError(
            "Number of labels does not match number of samples"
        )

    if len(flops_per_exit) != num_exits:
        raise ValueError(
            f"Model has {num_exits} exits but flops.json contains "
            f"{len(flops_per_exit)} FLOP values"
        )

    grid = make_threshold_grid(grid_step)

    evaluations = []

    print()
    print("Evaluating validation thresholds...")
    print(f"number of exits: {num_exits}")
    print(f"validation samples: {num_samples}")
    print(f"threshold candidates: {len(grid)}")
    print()

    for threshold in grid:
        # Use one shared confidence level across early exits.
        # The final exit is always the fallback.
        thresholds = (
            [float(threshold)] * (num_exits - 1)
            + [0.0]
        )

        result = evaluate_policy(
            logits,
            y,
            thresholds,
            flops_per_exit,
            method=method,
            temperature=temperature,
        )

        evaluations.append(result)

    full_flops = float(flops_per_exit[-1])

    selected: dict[str, list[float]] = {}

    report_lines = [
        "M5 VALIDATION THRESHOLD TUNING REPORT",
        "=" * 70,
        "",
        f"confidence method: {method}",
        f"temperature: {temperature}",
        f"validation samples: {num_samples}",
        f"exits: {num_exits}",
        f"FLOPs per exit: {flops_per_exit}",
        "",
    ]

    print("=" * 70)
    print("SELECTED BUDGET OPERATING POINTS")
    print("=" * 70)

    for budget_name, fraction in DEFAULT_BUDGETS.items():
        target = full_flops * fraction

        result, feasible = choose_for_budget(
            evaluations,
            target_flops=target,
        )

        selected[budget_name] = [
            float(v)
            for v in result["thresholds"]
        ]

        achieved_fraction = (
            result["mean_flops"] / full_flops
            if full_flops > 0
            else 0.0
        )

        status = "within target" if feasible else "target unattainable"

        line = (
            f"Budget {budget_name}% | "
            f"target <= {target:,.0f} FLOPs | "
            f"achieved {result['mean_flops']:,.0f} "
            f"({achieved_fraction * 100:.1f}% of full) | "
            f"bal_acc={result['bal_acc']:.4f} | "
            f"f1={result['f1']:.4f} | "
            f"acc={result['acc']:.4f} | "
            f"exits={result['exit_counts']} | "
            f"{status}"
        )

        print(line)
        print(f"  thresholds: {result['thresholds']}")
        print()

        report_lines.extend(
            [
                line,
                f"thresholds: {result['thresholds']}",
                "",
            ]
        )

    return selected, "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune M5 early-exit confidence thresholds using "
            "validation logits only."
        )
    )

    parser.add_argument(
        "--logits",
        required=True,
        help="Validation logits file produced by M3.",
    )

    parser.add_argument(
        "--flops",
        required=True,
        help="M4 flops.json file.",
    )

    parser.add_argument(
        "--output",
        default="thresholds.json",
        help="Output threshold JSON file.",
    )

    parser.add_argument(
        "--report",
        default="policy_tuning_report.txt",
        help="Human-readable tuning report.",
    )

    parser.add_argument(
        "--method",
        choices=["max_softmax", "entropy"],
        default="max_softmax",
        help="Confidence measure.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature.",
    )

    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.02,
        help="Threshold search step.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("M5 POLICY THRESHOLD TUNER")
    print("=" * 70)

    print(f"Loading validation logits: {args.logits}")
    logits, y = load_logits(args.logits)

    print(f"logits shape: {tuple(logits.shape)}")
    print(f"labels shape: {tuple(y.shape)}")

    print()
    print(f"Loading FLOPs profile: {args.flops}")
    flops_profile = load_flops(args.flops)

    if "flops_per_exit" not in flops_profile:
        raise ValueError(
            "flops.json does not contain 'flops_per_exit'"
        )

    flops_per_exit = [
        int(v)
        for v in flops_profile["flops_per_exit"]
    ]

    print(f"FLOPs per exit: {flops_per_exit}")

    thresholds, report = tune_thresholds(
        logits,
        y,
        flops_per_exit,
        method=args.method,
        temperature=args.temperature,
        grid_step=args.grid_step,
    )

    output_path = Path(args.output)
    report_path = Path(args.report)

    save_thresholds(
        output_path,
        thresholds,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 70)
    print("THRESHOLD TUNING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"thresholds: {output_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()