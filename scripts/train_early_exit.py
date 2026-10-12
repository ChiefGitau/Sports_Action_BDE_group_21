"""M3: train the early-exit model jointly over all exits, with knowledge distillation by
default, and dump per-exit logits for M5/M7.

    python scripts/train_early_exit.py                 # -> outputs/earlyexit/   (variant fp32+kd)
    python scripts/train_early_exit.py --no-kd         # ablation: joint training without KD
    python scripts/train_early_exit.py --no-kd --name earlyexit_nokd
"""
from __future__ import annotations

from _common import base_parser, prepare

NEAR_RANDOM = 0.55   # balanced accuracy below this on validation = "earliest exit is near random"


def main() -> None:
    p = base_parser(__doc__, default_name="earlyexit")
    p.add_argument("--no-kd", action="store_true", help="disable knowledge distillation")
    p.add_argument("--kd-alpha", type=float, default=None, help="override early_exit.kd_alpha")
    p.add_argument("--kd-temperature", type=float, default=None, help="override early_exit.kd_temperature")
    args = p.parse_args()
    cfg = prepare(args)

    import models

    ee = cfg.early_exit
    use_kd = bool(ee.kd) and not args.no_kd
    kd_alpha = (ee.kd_alpha if args.kd_alpha is None else args.kd_alpha) if use_kd else 0.0
    kd_T = ee.kd_temperature if args.kd_temperature is None else args.kd_temperature
    variant = "fp32+kd" if use_kd else "fp32"

    splits = models.load_splits(cfg)
    arch = models.arch_from_config(cfg, "early_exit")
    model = models.build_model(cfg, "early_exit")
    weights = list(ee.exit_loss_weights)
    if len(weights) != model.num_exits:
        raise SystemExit(f"early_exit.exit_loss_weights has {len(weights)} entries but model has {model.num_exits} exits")
    print(f"EarlyExitCNN exits after blocks {model.exit_after}  widths {model.channel_widths()}  "
          f"variant={variant} kd_alpha={kd_alpha} T={kd_T} exit_weights={weights}  device={models.get_device(cfg)}")

    models.fit(model, splits["train"], splits["val"], cfg,
               exit_weights=weights, kd_alpha=kd_alpha, kd_temperature=kd_T, epochs=args.epochs, lr=args.lr)

    val_metrics = models.evaluate(model, splits["val"], cfg)
    print("\nper-exit validation metrics:")
    for e, m in enumerate(val_metrics):
        flag = "   <-- near random; consider moving this exit deeper" if m["bal_acc"] < NEAR_RANDOM else ""
        print(f"  exit{e} (after block {model.exit_after[e]}): acc {m['acc']:.3f}  bal_acc {m['bal_acc']:.3f}  f1 {m['f1']:.3f}{flag}")

    if not args.no_export:
        models.export(model, args.name, variant, cfg, splits, kind="early_exit", arch=arch)


if __name__ == "__main__":
    main()
