"""M2: train the full static baseline (single exit) and export it.

    python scripts/train_baseline.py                 # -> outputs/cnn_full/
    python scripts/train_baseline.py --epochs 5      # quick check on mock data
"""
from __future__ import annotations

from _common import base_parser, prepare


def main() -> None:
    args = base_parser(__doc__, default_name="cnn_full").parse_args()
    cfg = prepare(args)

    import models

    splits = models.load_splits(cfg)
    print(f"train {tuple(splits['train']['X'].shape)}  val {tuple(splits['val']['X'].shape)}  "
          f"test {tuple(splits['test']['X'].shape)}  device={models.get_device(cfg)}")

    arch = models.arch_from_config(cfg, "static")
    model = models.build_model(cfg, "static")
    print(f"StaticCNN widths {model.channel_widths()}  params {sum(p.numel() for p in model.parameters())}")

    models.fit(model, splits["train"], splits["val"], cfg, epochs=args.epochs, lr=args.lr)
    if not args.no_export:
        models.export(model, args.name, "fp32", cfg, splits, kind="static", arch=arch)


if __name__ == "__main__":
    main()
