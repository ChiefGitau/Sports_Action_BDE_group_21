"""M0 'done when' check: mock data loads, a dummy model runs through the interface,
and one row lands in results.csv. Also writes example logits, flops.json and thresholds.json
so M4/M5/M7 have real files in the right format to develop against.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core  # noqa: E402


def main() -> None:
    cfg = core.load_config()
    core.set_seed(cfg.seed)

    # 1. Mock data in the contract format.
    paths = core.write_mock_dataset(cfg)
    val = core.load_split(paths["val"], num_classes=cfg.data.num_classes)
    N, T, P, _ = val["X"].shape
    print(f"mock val split: X={tuple(val['X'].shape)}  class balance={core.class_balance(val['y'], cfg.data.num_classes)}")

    # 2. Dummy model through the shared interface.
    model = core.DummyExitModel(
        max_people=P, num_classes=cfg.data.num_classes, num_exits=cfg.model.num_exits, hidden=cfg.model.hidden
    ).eval()
    with torch.no_grad():
        outs = model(val["X"], val["mask"])
    logits = core.stack_exit_logits(outs)                    # [E, N, C]
    print(f"model: {model.num_exits} exits, logits={tuple(logits.shape)}")

    # 3. Cost profile and metrics per exit.
    prof = core.profile(model, val["X"], val["mask"])
    print(f"flops per exit ({prof['tool']}): {prof['flops_per_exit']}  params={prof['params']}  size={prof['size_mb']:.3f} MB")

    # 4. Artifacts in the agreed formats.
    out_dir = core.resolve_path(cfg, "outputs") / "dummy"
    core.save_logits(out_dir / "logits_val.pt", logits, val["y"])
    core.save_flops(out_dir / "flops.json", prof)
    core.save_thresholds(
        out_dir / "thresholds.json", {lvl: [0.9] * (model.num_exits - 1) + [0.0] for lvl in cfg.budget.levels}
    )

    results_csv = core.resolve_path(cfg, "results_csv")
    for e in range(model.num_exits):
        m = core.compute_metrics(logits[e], val["y"], cfg.data.num_classes)
        core.append_result(
            results_csv,
            core.ResultRow(
                model="dummy", variant=f"exit{e}", budget="static", split="val",
                flops_mean=prof["flops_per_exit"][e], size_mb=prof["size_mb"], **m,
            ),
        )
    df = core.load_results(results_csv)
    print(f"results.csv now has {len(df)} rows -> {results_csv}")
    print(df.tail(model.num_exits).to_string(index=False))
    print("\nM0 smoke test passed.")


if __name__ == "__main__":
    main()
