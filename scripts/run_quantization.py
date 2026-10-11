from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from torch import Tensor, nn

from compression.quantization import (
    is_actually_quantized,
    quantize_ptq_fx,
    quantized_module_names,
    select_calibration_indices,
)
from core.artifacts import ResultRow, append_result
from core.data import load_split
from core.flops import flops_per_exit
from core.metrics import compute_metrics
from core.model import DummyExitModel


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized state_dict size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def load_checkpoint_object(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(
    checkpoint: Path | None,
    max_people: int,
    num_classes: int,
) -> tuple[nn.Module, bool]:
    """
    Return (model, is_dummy).

    Until M3 is integrated, running without --checkpoint uses the Core
    DummyExitModel only for an M6 smoke test.

    If the M3 checkpoint contains a complete nn.Module, this script can use
    it directly. If M3 supplies only a state_dict, its exact architecture
    constructor must be connected here after M3 is merged.
    """
    if checkpoint is None:
        model = DummyExitModel(
            max_people=max_people,
            num_classes=num_classes,
            num_exits=4,
        )
        return model.eval(), True

    obj = load_checkpoint_object(checkpoint)

    if isinstance(obj, nn.Module):
        return obj.cpu().eval(), False

    if isinstance(obj, dict):
        for key in ("model", "module"):
            candidate = obj.get(key)
            if isinstance(candidate, nn.Module):
                return candidate.cpu().eval(), False

    raise RuntimeError(
        "The checkpoint does not contain a complete nn.Module.\n"
        "This is not necessarily an error in M3: the teammate may be saving "
        "only a state_dict. Once the final M3 checkpoint format is known, "
        "connect its model constructor inside load_model()."
    )


def final_exit_logits(
    model: nn.Module,
    X: Tensor,
    mask: Tensor,
    batch_size: int,
    quantized: bool,
) -> Tensor:
    """Run the model and collect logits from the deepest exit."""
    model.eval()
    chunks: list[Tensor] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start : start + batch_size].cpu().float()
            mb = mask[start : start + batch_size].cpu()

            # FX INT8 observers/quantized graphs need floating tensors.
            # The project data contract itself remains bool.
            if quantized:
                mb = mb.to(dtype=xb.dtype)

            outputs = model(xb, mb)

            if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
                raise RuntimeError(
                    "Expected model.forward() to return one tensor per exit."
                )

            chunks.append(outputs[-1].detach().cpu())

    return torch.cat(chunks, dim=0)


def evaluate(
    model: nn.Module,
    X: Tensor,
    mask: Tensor,
    y: Tensor,
    batch_size: int,
    num_classes: int,
    quantized: bool,
) -> dict[str, float]:
    logits = final_exit_logits(
        model,
        X,
        mask,
        batch_size=batch_size,
        quantized=quantized,
    )
    return compute_metrics(logits, y.cpu(), num_classes=num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M6 post-training INT8 quantization."
    )

    parser.add_argument(
        "--train",
        default="data/processed/train.pt",
        help="Training split used only to choose calibration samples.",
    )
    parser.add_argument(
        "--eval",
        default="data/processed/val.pt",
        help="Split used for FP32 vs INT8 accuracy comparison.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Final M3 early-exit checkpoint. Omit for DummyExitModel smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/m6",
        help="Directory for the INT8 checkpoint and report.",
    )
    parser.add_argument(
        "--results",
        default="results.csv",
        help="Shared project results.csv.",
    )
    parser.add_argument(
        "--max-calibration",
        type=int,
        default=256,
        help="Maximum number of calibration windows.",
    )
    parser.add_argument(
        "--min-action",
        type=int,
        default=32,
        help="Desired minimum number of action windows in calibration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--backend",
        default="onednn",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    eval_path = Path(args.eval)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    print("=" * 72)
    print("M6 POST-TRAINING INT8 QUANTIZATION")
    print("=" * 72)

    print(f"\nLoading training split: {train_path}")
    train = load_split(train_path)

    print(f"Loading evaluation split: {eval_path}")
    evaluation = load_split(eval_path)

    X_train = train["X"]
    mask_train = train["mask"]
    y_train = train["y"]

    X_eval = evaluation["X"]
    mask_eval = evaluation["mask"]
    y_eval = evaluation["y"]

    max_people = int(X_train.shape[2])

    all_labels = torch.cat([y_train.cpu(), y_eval.cpu()])
    num_classes = max(2, int(all_labels.max().item()) + 1)

    print(f"training windows: {len(y_train)}")
    print(f"evaluation windows: {len(y_eval)}")
    print(f"max people: {max_people}")
    print(f"classes: {num_classes}")

    action_total = int((y_train == 1).sum().item())
    print(f"available action windows for calibration: {action_total}")

    if action_total == 0:
        raise RuntimeError(
            "M6 requires action windows in the calibration data, "
            "but the training split contains none."
        )

    calibration_indices = select_calibration_indices(
        y_train,
        max_samples=args.max_calibration,
        min_action_samples=args.min_action,
        seed=args.seed,
    )

    calibration_x = X_train[calibration_indices]
    calibration_mask = mask_train[calibration_indices]
    calibration_y = y_train[calibration_indices]

    calibration_actions = int((calibration_y == 1).sum().item())

    print(f"selected calibration windows: {len(calibration_indices)}")
    print(f"selected action windows: {calibration_actions}")

    if calibration_actions == 0:
        raise RuntimeError(
            "Calibration selection produced no action windows."
        )

    fp32_model, dummy_mode = load_model(
        checkpoint_path,
        max_people=max_people,
        num_classes=num_classes,
    )

    if dummy_mode:
        print("\nWARNING: using DummyExitModel.")
        print("This run checks M6 machinery only; it is NOT a final experiment.")
        output_dir = Path(".m6_smoke")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating FP32 model...")
    fp32_metrics = evaluate(
        fp32_model,
        X_eval,
        mask_eval,
        y_eval,
        batch_size=args.batch_size,
        num_classes=num_classes,
        quantized=False,
    )

    fp32_size = state_dict_size_mb(fp32_model)

    print("\nMeasuring FP32 FLOPs...")
    try:
        exit_flops = flops_per_exit(
            fp32_model,
            X_eval[:1],
            mask_eval[:1],
        )
        full_flops = float(exit_flops[-1])
        print(f"FLOPs per exit: {exit_flops}")
    except Exception as exc:
        print(f"WARNING: FLOPs measurement failed: {exc}")
        full_flops = float("nan")

    print("\nRunning post-training INT8 quantization...")

    int8_model = quantize_ptq_fx(
        fp32_model,
        calibration_x,
        calibration_mask,
        backend=args.backend,
        batch_size=args.batch_size,
    )

    if not is_actually_quantized(int8_model):
        raise RuntimeError(
            "FX conversion completed but no quantized modules were found."
        )

    quantized_names = quantized_module_names(int8_model)

    print(f"quantized modules found: {len(quantized_names)}")

    print("\nEvaluating INT8 model...")
    int8_metrics = evaluate(
        int8_model,
        X_eval,
        mask_eval,
        y_eval,
        batch_size=args.batch_size,
        num_classes=num_classes,
        quantized=True,
    )

    int8_size = state_dict_size_mb(int8_model)

    checkpoint_output = output_dir / "earlyexit_int8.pt"

    print(f"\nSaving INT8 checkpoint: {checkpoint_output}")
    torch.save(
        {
            "model": int8_model,
            "variant": "int8",
            "backend": args.backend,
            "calibration_windows": len(calibration_indices),
            "calibration_action_windows": calibration_actions,
        },
        checkpoint_output,
    )

    split_name = eval_path.stem

    report_lines = [
        "M6 INT8 QUANTIZATION REPORT",
        "=" * 72,
        f"dummy_mode: {dummy_mode}",
        f"evaluation_split: {split_name}",
        f"backend: {args.backend}",
        f"calibration_windows: {len(calibration_indices)}",
        f"calibration_action_windows: {calibration_actions}",
        "",
        "FP32",
        f"  acc: {fp32_metrics['acc']:.6f}",
        f"  bal_acc: {fp32_metrics['bal_acc']:.6f}",
        f"  f1: {fp32_metrics['f1']:.6f}",
        f"  size_mb: {fp32_size:.6f}",
        "",
        "INT8",
        f"  acc: {int8_metrics['acc']:.6f}",
        f"  bal_acc: {int8_metrics['bal_acc']:.6f}",
        f"  f1: {int8_metrics['f1']:.6f}",
        f"  size_mb: {int8_size:.6f}",
        "",
        f"size_reduction_percent: {(1.0 - int8_size / fp32_size) * 100.0:.2f}",
        f"quantized_modules: {len(quantized_names)}",
    ]

    report = "\n".join(report_lines)

    report_path = output_dir / "quantization_report.txt"
    report_path.write_text(report)

    print()
    print(report)

    if dummy_mode:
        print("\nDummy smoke test: results.csv was NOT modified.")
    else:
        results_path = Path(args.results)

        append_result(
            results_path,
            ResultRow(
                model="earlyexit",
                variant="fp32",
                budget="100",
                split=split_name,
                flops_mean=full_flops,
                acc=fp32_metrics["acc"],
                bal_acc=fp32_metrics["bal_acc"],
                f1=fp32_metrics["f1"],
                size_mb=fp32_size,
            ),
        )

        append_result(
            results_path,
            ResultRow(
                model="earlyexit",
                variant="int8",
                budget="100",
                split=split_name,
                flops_mean=full_flops,
                acc=int8_metrics["acc"],
                bal_acc=int8_metrics["bal_acc"],
                f1=int8_metrics["f1"],
                size_mb=int8_size,
            ),
        )

        print(f"\nAppended FP32 and INT8 rows to: {results_path}")

    print("\n" + "=" * 72)
    print("M6 QUANTIZATION RUN COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()