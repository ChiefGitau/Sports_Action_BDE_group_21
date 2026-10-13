"""M7: final evaluation of static, adaptive, pruned, and INT8 models.

Important experimental rule:
    thresholds are selected on the validation split in M5 and are only
    applied here to the test split. M7 never tunes on test data.

Outputs:
    outputs/final/results.csv
    outputs/final/pareto_front.csv
    outputs/final/policy_details.json
    outputs/final/evaluation_report.txt
    outputs/final/balanced_accuracy_vs_flops.png
    outputs/final/f1_vs_flops.png
    outputs/final/balanced_accuracy_vs_size.png

The script also creates outputs/earlyexit_int8/logits_test.pt so the
quantized test predictions do not need to be recomputed unnecessarily.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from core.artifacts import RESULT_COLUMNS, load_flops, load_logits
from core.metrics import compute_metrics
from core.policy import (
    apply_threshold_policy,
    exit_histogram,
    mean_flops,
)


ROOT = Path(__file__).resolve().parents[1]

TEST_SPLIT = ROOT / "data" / "processed" / "test.pt"

CNN_FULL_DIR = ROOT / "outputs" / "cnn_full"
CNN_PRUNED_DIR = ROOT / "outputs" / "cnn_pruned"
EARLYEXIT_DIR = ROOT / "outputs" / "earlyexit"
INT8_DIR = ROOT / "outputs" / "earlyexit_int8"


def load_test_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the canonical M1 test split."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    X = torch.as_tensor(data["X"]).float()
    mask = torch.as_tensor(data["mask"])
    y = torch.as_tensor(data["y"]).long()

    return X, mask, y


def check_labels(reference: torch.Tensor, candidate: torch.Tensor, name: str) -> None:
    """Prevent accidental comparison of models evaluated on different splits."""
    reference = reference.cpu().long()
    candidate = candidate.cpu().long()

    if reference.shape != candidate.shape:
        raise ValueError(
            f"{name}: label shape {tuple(candidate.shape)} does not match "
            f"canonical test labels {tuple(reference.shape)}"
        )

    if not torch.equal(reference, candidate):
        raise ValueError(
            f"{name}: labels differ from the canonical test split. "
            "Refusing to combine incompatible experiments."
        )


def find_threshold_map(obj) -> dict[str, list[float]] | None:
    """Find the 100/50/20 threshold dictionary in an M5 JSON artifact."""
    if not isinstance(obj, dict):
        return None

    candidate = {}

    for key in ("100", "50", "20"):
        value = obj.get(key)

        if isinstance(value, list):
            candidate[key] = [float(v) for v in value]

    if candidate:
        return candidate

    # Be tolerant of metadata wrappers used by different artifact versions.
    for value in obj.values():
        if isinstance(value, dict):
            found = find_threshold_map(value)
            if found is not None:
                return found

    return None


def load_thresholds(path: Path) -> dict[str, list[float]]:
    """Load thresholds selected on validation data."""
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    thresholds = find_threshold_map(obj)

    if thresholds is None:
        raise ValueError(f"Could not find budget thresholds in {path}")

    for required in ("100", "50", "20"):
        if required not in thresholds:
            raise ValueError(f"{path} is missing budget {required}")

    return thresholds


def normalise_model_output(output) -> torch.Tensor:
    """Convert TorchScript early-exit output to [E, B, C]."""
    if isinstance(output, torch.Tensor):
        if output.ndim == 2:
            return output.unsqueeze(0)

        if output.ndim == 3:
            return output

        raise ValueError(
            f"Unexpected TorchScript tensor shape: {tuple(output.shape)}"
        )

    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("INT8 model returned no exits.")

        return torch.stack(
            [torch.as_tensor(item) for item in output],
            dim=0,
        )

    raise TypeError(
        f"Unexpected INT8 model output type: {type(output).__name__}"
    )


def generate_int8_test_logits(
    checkpoint: Path,
    X: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.jit.ScriptModule]:
    """Run the reloadable M6 INT8 model on the canonical M1 test split."""

    model = torch.jit.load(str(checkpoint), map_location="cpu")
    model.eval()

    batches = []

    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))

            xb = X[start:end].float()

            # M6's FX quantization uses a float mask because quantization
            # observers cannot process the original boolean mask directly.
            mb = mask[start:end].float()

            output = model(xb, mb)
            stacked = normalise_model_output(output).cpu()

            batches.append(stacked)

    logits = torch.cat(batches, dim=1)

    if logits.shape[1] != len(y):
        raise RuntimeError(
            f"INT8 inference produced {logits.shape[1]} samples, "
            f"expected {len(y)}."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "logits": logits,
            "y": y.cpu(),
        },
        cache_path,
    )

    return logits, y.cpu(), model


def load_int8_size_mb(report_path: Path) -> float:
    """Load the INT8 model-size measurement produced during M6.

    M6 measures FP32 and INT8 model size using the same serialization
    method, making this value appropriate for the final comparison.
    The deployable TorchScript file can be larger because it also stores
    graph/runtime metadata.
    """
    lines = report_path.read_text(encoding="utf-8").splitlines()

    in_int8_section = False

    for line in lines:
        stripped = line.strip()

        if stripped == "INT8":
            in_int8_section = True
            continue

        if in_int8_section and stripped.startswith("size_mb:"):
            return float(stripped.split(":", 1)[1].strip())

    raise ValueError(
        f"Could not find INT8 size_mb in {report_path}"
    )


def static_metrics(logits: torch.Tensor, y: torch.Tensor, exit_id: int) -> dict:
    """Evaluate one fixed exit."""
    metrics = compute_metrics(
        logits[exit_id].cpu(),
        y.cpu(),
    )

    return {
        "acc": float(metrics["acc"]),
        "bal_acc": float(metrics["bal_acc"]),
        "f1": float(metrics["f1"]),
    }


def adaptive_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    thresholds: list[float],
    flops_per_exit: list[int],
) -> dict:
    """Apply validation-selected thresholds to test logits."""
    _, exit_ids, _ = apply_threshold_policy(
        logits,
        thresholds,
        method="max_softmax",
        temperature=1.0,
    )

    sample_ids = torch.arange(logits.shape[1])

    selected_logits = logits[
        exit_ids,
        sample_ids,
        :,
    ]

    metrics = compute_metrics(
        selected_logits.cpu(),
        y.cpu(),
    )

    return {
        "acc": float(metrics["acc"]),
        "bal_acc": float(metrics["bal_acc"]),
        "f1": float(metrics["f1"]),
        "mean_flops": float(
            mean_flops(
                exit_ids,
                flops_per_exit,
            )
        ),
        "exit_counts": exit_histogram(
            exit_ids,
            logits.shape[0],
        ),
    }


def add_result(
    rows: list[dict],
    *,
    model: str,
    variant: str,
    budget: str,
    flops: float,
    metrics: dict,
    size_mb: float,
) -> None:
    """Append one operating point using the shared ResultRow schema."""
    rows.append(
        {
            "model": model,
            "variant": variant,
            "budget": budget,
            "split": "test",
            "flops_mean": float(flops),
            "acc": float(metrics["acc"]),
            "bal_acc": float(metrics["bal_acc"]),
            "f1": float(metrics["f1"]),
            "size_mb": float(size_mb),
            "latency_ms": float("nan"),
        }
    )


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return points not dominated in FLOPs vs balanced accuracy.

    A point is dominated if another point uses no more FLOPs and has
    at least as high balanced accuracy, with one strict improvement.
    """
    keep = []

    for i, row in df.iterrows():
        dominated = False

        for j, other in df.iterrows():
            if i == j:
                continue

            no_more_compute = other["flops_mean"] <= row["flops_mean"]
            no_worse_quality = other["bal_acc"] >= row["bal_acc"]

            strictly_better = (
                other["flops_mean"] < row["flops_mean"]
                or other["bal_acc"] > row["bal_acc"]
            )

            if no_more_compute and no_worse_quality and strictly_better:
                dominated = True
                break

        keep.append(not dominated)

    return df.loc[keep].copy()


def short_label(row: pd.Series) -> str:
    """Compact labels for figures."""
    if row["model"] == "cnn_full":
        return "Full CNN"

    if row["model"] == "cnn_pruned":
        return "Pruned CNN"

    if row["variant"] == "fp32/final_exit":
        return "EE FP32 full"

    if row["variant"] == "int8/final_exit":
        return "EE INT8 full"

    precision = "FP32" if row["variant"].startswith("fp32") else "INT8"
    return f"{precision} policy {row['budget']}%"


def make_plot(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    output: Path,
) -> None:
    """Create a simple comparison figure without hiding individual points."""
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df[x_column],
        df[y_column],
        s=60,
    )

    for _, row in df.iterrows():
        plt.annotate(
            short_label(row),
            (row[x_column], row[y_column]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M7: canonical final test-set evaluation."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for INT8 test inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final",
        help="Directory for final M7 artifacts.",
    )

    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("M7 FINAL EVALUATION")
    print("=" * 72)

    X_test, mask_test, y_test = load_test_split(TEST_SPLIT)

    print(f"canonical test windows: {len(y_test)}")
    print(f"X: {tuple(X_test.shape)}")
    print(f"mask: {tuple(mask_test.shape)}")

    # ------------------------------------------------------------------
    # M2 static baselines
    # ------------------------------------------------------------------

    full_logits, full_y = load_logits(
        CNN_FULL_DIR / "logits_test.pt"
    )
    pruned_logits, pruned_y = load_logits(
        CNN_PRUNED_DIR / "logits_test.pt"
    )

    check_labels(y_test, full_y, "cnn_full")
    check_labels(y_test, pruned_y, "cnn_pruned")

    full_flops_meta = load_flops(
        CNN_FULL_DIR / "flops.json"
    )
    pruned_flops_meta = load_flops(
        CNN_PRUNED_DIR / "flops.json"
    )

    full_flops = full_flops_meta["flops_per_exit"]
    pruned_flops = pruned_flops_meta["flops_per_exit"]

    # ------------------------------------------------------------------
    # M3 FP32 early-exit model
    # ------------------------------------------------------------------

    fp32_logits, fp32_y = load_logits(
        EARLYEXIT_DIR / "logits_test.pt"
    )

    check_labels(y_test, fp32_y, "earlyexit FP32")

    fp32_flops_meta = load_flops(
        EARLYEXIT_DIR / "flops.json"
    )
    fp32_flops = fp32_flops_meta["flops_per_exit"]

    fp32_thresholds = load_thresholds(
        EARLYEXIT_DIR / "thresholds.json"
    )

    # ------------------------------------------------------------------
    # M6 INT8 early-exit model
    # ------------------------------------------------------------------

    int8_checkpoint = INT8_DIR / "earlyexit_int8.pt"

    print("\nRunning INT8 model on canonical test split...")

    int8_logits, int8_y, int8_model = generate_int8_test_logits(
        int8_checkpoint,
        X_test,
        mask_test,
        y_test,
        batch_size=args.batch_size,
        cache_path=INT8_DIR / "logits_test.pt",
    )

    check_labels(y_test, int8_y, "earlyexit INT8")

    if int8_logits.shape[0] != len(fp32_flops):
        raise ValueError(
            f"INT8 model has {int8_logits.shape[0]} exits but "
            f"FP32 FLOP metadata contains {len(fp32_flops)} exits."
        )

    int8_thresholds = load_thresholds(
        INT8_DIR / "thresholds.json"
    )

    int8_size_mb = load_int8_size_mb(
        INT8_DIR / "quantization_report.txt"
    )

    # ------------------------------------------------------------------
    # Final test operating points
    # ------------------------------------------------------------------

    rows: list[dict] = []
    policy_details: dict[str, dict] = {}

    full_metrics = static_metrics(
        full_logits,
        full_y,
        0,
    )

    add_result(
        rows,
        model="cnn_full",
        variant="fp32",
        budget="static",
        flops=full_flops[0],
        metrics=full_metrics,
        size_mb=full_flops_meta["size_mb"],
    )

    pruned_metrics = static_metrics(
        pruned_logits,
        pruned_y,
        0,
    )

    add_result(
        rows,
        model="cnn_pruned",
        variant="fp32+pruned",
        budget="static",
        flops=pruned_flops[0],
        metrics=pruned_metrics,
        size_mb=pruned_flops_meta["size_mb"],
    )

    fp32_final_metrics = static_metrics(
        fp32_logits,
        fp32_y,
        fp32_logits.shape[0] - 1,
    )

    add_result(
        rows,
        model="earlyexit",
        variant="fp32/final_exit",
        budget="static",
        flops=fp32_flops[-1],
        metrics=fp32_final_metrics,
        size_mb=fp32_flops_meta["size_mb"],
    )

    int8_final_metrics = static_metrics(
        int8_logits,
        int8_y,
        int8_logits.shape[0] - 1,
    )

    add_result(
        rows,
        model="earlyexit",
        variant="int8/final_exit",
        budget="static",
        flops=fp32_flops[-1],
        metrics=int8_final_metrics,
        size_mb=int8_size_mb,
    )

    full_ee_flops = float(fp32_flops[-1])
    min_ee_flops = float(fp32_flops[0])

    budget_ratios = {
        "100": 1.00,
        "50": 0.50,
        "20": 0.20,
    }

    for precision, logits, labels, thresholds_map, variant, size_mb in (
        (
            "fp32",
            fp32_logits,
            fp32_y,
            fp32_thresholds,
            "fp32/adaptive",
            fp32_flops_meta["size_mb"],
        ),
        (
            "int8",
            int8_logits,
            int8_y,
            int8_thresholds,
            "int8/adaptive",
            int8_size_mb,
        ),
    ):
        for budget in ("100", "50", "20"):
            thresholds = thresholds_map[budget]

            result = adaptive_metrics(
                logits,
                labels,
                thresholds,
                fp32_flops,
            )

            add_result(
                rows,
                model="earlyexit",
                variant=variant,
                budget=budget,
                flops=result["mean_flops"],
                metrics=result,
                size_mb=size_mb,
            )

            target_flops = (
                full_ee_flops * budget_ratios[budget]
            )

            policy_details[f"{precision}_{budget}"] = {
                "thresholds": thresholds,
                "target_flops": target_flops,
                "mean_test_flops": result["mean_flops"],
                "compute_fraction_of_full": (
                    result["mean_flops"] / full_ee_flops
                ),
                "exit_counts": result["exit_counts"],
                "architecturally_feasible": (
                    min_ee_flops <= target_flops + 1e-9
                ),
                "threshold_source": "validation",
                "evaluation_split": "test",
            }

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    results_path = output_dir / "results.csv"
    df.to_csv(results_path, index=False)

    pareto = pareto_front(df)

    pareto_path = output_dir / "pareto_front.csv"
    pareto.to_csv(pareto_path, index=False)

    details_path = output_dir / "policy_details.json"

    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(
            policy_details,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    plot_df = df.copy()
    plot_df["flops_millions"] = (
        plot_df["flops_mean"] / 1_000_000
    )

    make_plot(
        plot_df,
        x_column="flops_millions",
        y_column="bal_acc",
        x_label="Mean compute (million FLOPs / sample)",
        y_label="Balanced accuracy",
        output=output_dir / "balanced_accuracy_vs_flops.png",
    )

    make_plot(
        plot_df,
        x_column="flops_millions",
        y_column="f1",
        x_label="Mean compute (million FLOPs / sample)",
        y_label="F1 score",
        output=output_dir / "f1_vs_flops.png",
    )

    make_plot(
        plot_df,
        x_column="size_mb",
        y_column="bal_acc",
        x_label="Serialized model size (MB)",
        y_label="Balanced accuracy",
        output=output_dir / "balanced_accuracy_vs_size.png",
    )

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    report_lines = [
        "M7 FINAL EVALUATION REPORT",
        "=" * 72,
        "",
        f"Canonical test windows: {len(y_test)}",
        "",
        "Methodological rule:",
        "  Policy thresholds were selected on validation data in M5.",
        "  M7 applies those fixed thresholds to test data without retuning.",
        "",
        "FINAL TEST RESULTS",
        "-" * 72,
    ]

    for _, row in df.iterrows():
        report_lines.append(
            f"{row['model']:12s} "
            f"{row['variant']:18s} "
            f"budget={str(row['budget']):6s} "
            f"FLOPs={row['flops_mean']:10.0f} "
            f"acc={row['acc']:.4f} "
            f"bal_acc={row['bal_acc']:.4f} "
            f"f1={row['f1']:.4f} "
            f"size={row['size_mb']:.4f} MB"
        )

    report_lines.extend(
        [
            "",
            "ADAPTIVE POLICY DETAILS",
            "-" * 72,
        ]
    )

    for name, details in policy_details.items():
        feasible = details["architecturally_feasible"]

        report_lines.append(
            f"{name}: thresholds={details['thresholds']} "
            f"mean_flops={details['mean_test_flops']:.0f} "
            f"fraction={100 * details['compute_fraction_of_full']:.1f}% "
            f"exit_counts={details['exit_counts']} "
            f"architecture_feasible={feasible}"
        )

    report_lines.extend(
        [
            "",
            "PARETO FRONT (balanced accuracy vs FLOPs)",
            "-" * 72,
        ]
    )

    for _, row in pareto.iterrows():
        report_lines.append(
            f"{row['model']} / {row['variant']} / {row['budget']}: "
            f"{row['flops_mean']:.0f} FLOPs, "
            f"bal_acc={row['bal_acc']:.4f}"
        )

    minimum_fraction = min_ee_flops / full_ee_flops

    report_lines.extend(
        [
            "",
            "BUDGET NOTE",
            "-" * 72,
            (
                f"The first early exit costs {min_ee_flops:.0f} FLOPs, "
                f"which is {100 * minimum_fraction:.1f}% of full-model compute."
            ),
            (
                "Therefore a true 20% early-exit compute budget is "
                "architecturally impossible for this network."
            ),
            "",
            "INT8 NOTE",
            "-" * 72,
            (
                "INT8 uses separately validation-tuned confidence thresholds. "
                "The quantized model is not evaluated with FP32 thresholds."
            ),
        ]
    )

    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL TEST RESULTS")
    print(
        df[
            [
                "model",
                "variant",
                "budget",
                "flops_mean",
                "acc",
                "bal_acc",
                "f1",
                "size_mb",
            ]
        ].to_string(index=False)
    )

    print("\nPARETO FRONT")
    print(
        pareto[
            [
                "model",
                "variant",
                "budget",
                "flops_mean",
                "bal_acc",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\nWROTE")
    print(f"  {results_path}")
    print(f"  {pareto_path}")
    print(f"  {details_path}")
    print(f"  {report_path}")
    print(f"  {output_dir / 'balanced_accuracy_vs_flops.png'}")
    print(f"  {output_dir / 'f1_vs_flops.png'}")
    print(f"  {output_dir / 'balanced_accuracy_vs_size.png'}")


if __name__ == "__main__":
    main()
