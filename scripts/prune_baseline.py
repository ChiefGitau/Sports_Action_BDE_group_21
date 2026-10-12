"""M2: physically prune the trained full baseline, fine-tune, export as the pruned baseline.

    python scripts/train_baseline.py                       # first
    python scripts/prune_baseline.py                       # -> outputs/cnn_pruned/
    python scripts/prune_baseline.py --ratio 0.25 --name cnn_pruned25
"""
from __future__ import annotations

from _common import base_parser, prepare


def main() -> None:
    p = base_parser(__doc__, default_name="cnn_pruned")
    p.add_argument("--source", default="cnn_full", help="outputs/<source>/model.pt to prune")
    p.add_argument("--ratio", type=float, default=None, help="override prune.ratio")
    args = p.parse_args()
    cfg = prepare(args)

    import core
    import models

    ratio = cfg.prune.ratio if args.ratio is None else args.ratio
    epochs = cfg.prune.finetune_epochs if args.epochs is None else args.epochs

    src = core.resolve_path(cfg, "outputs") / args.source / "model.pt"
    model, meta = models.load_checkpoint(src)
    if model.num_exits != 1:
        raise SystemExit(f"{src} has {model.num_exits} exits; prune_baseline.py expects the static baseline")
    splits = models.load_splits(cfg)
    val = splits["val"]

    before = core.profile(model, val["X"][:1], val["mask"][:1])
    print(f"loaded {src}: widths {model.channel_widths()}  flops {before['flops_per_exit'][0]}  params {before['params']}")
    print("val before pruning:", models.evaluate(model, val, cfg, device=__import__('torch').device('cpu'))[0])

    models.prune_channels(model, val["X"], val["mask"], ratio)
    after = core.profile(model, val["X"][:1], val["mask"][:1])
    print(f"pruned ratio={ratio}: widths {model.channel_widths()}  flops {after['flops_per_exit'][0]}  params {after['params']}")
    print("val right after pruning (no fine-tune):", models.evaluate(model, val, cfg, device=__import__('torch').device('cpu'))[0])

    models.fit(model, splits["train"], val, cfg, epochs=epochs, lr=args.lr)
    if not args.no_export:
        models.export(model, args.name, "fp32", cfg, splits, kind="static", arch=meta["arch"], pruned=True)


if __name__ == "__main__":
    main()
