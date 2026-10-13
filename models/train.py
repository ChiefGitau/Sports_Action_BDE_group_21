"""Training, evaluation and export shared by M2 and M3.

    fit(model, train, val, cfg, ...)   joint training over all exits, optional KD, best-epoch
                                        selection on the validation `select_metric` of the final exit
    predict(model, split, cfg)         -> logits [E, N, C] on one split
    export(model, name, variant, cfg)  -> outputs/{name}/{model.pt, flops.json, logits_val.pt,
                                           logits_test.pt} + one results.csv row per exit and split

Loss for a model with exits e = 0..E-1 (E = 1 for a static baseline):

    L = sum_e w_e * [ (1 - a_e) * CE(z_e, y)  +  a_e * T^2 * KL(softmax(z_E-1 / T) || softmax(z_e / T)) ]

with a_e = kd_alpha for early exits and 0 for the final exit, whose soft targets are detached.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import core
from core.config import AttrDict

from .cnn import ExitCNN
from .registry import save_checkpoint


# ---------- helpers ----------

def get_device(cfg: AttrDict) -> torch.device:
    want = str(cfg.train.get("device", "auto"))
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(y: Tensor, num_classes: int, mode: str) -> Tensor | None:
    """Inverse-frequency weights for cross-entropy ('balanced'), or None."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"train.class_weight must be 'balanced' or 'none', got {mode!r}")
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1)
    w = counts.sum() / (num_classes * counts)
    return w


def exit_loss(
    outs: list[Tensor],
    y: Tensor,
    ce_weight: Tensor | None,
    exit_weights: list[float],
    kd_alpha: float,
    kd_temperature: float,
) -> tuple[Tensor, dict[str, float]]:
    if len(exit_weights) != len(outs):
        raise ValueError(f"need one exit weight per exit: {len(exit_weights)} vs {len(outs)}")
    final = outs[-1]
    T = kd_temperature
    soft_target = F.softmax(final.detach() / T, dim=-1)
    total = outs[0].new_zeros(())
    parts: dict[str, float] = {}
    for e, (z, w) in enumerate(zip(outs, exit_weights)):
        ce = F.cross_entropy(z, y, weight=ce_weight)
        if e < len(outs) - 1 and kd_alpha > 0:
            kd = F.kl_div(F.log_softmax(z / T, dim=-1), soft_target, reduction="batchmean") * T * T
            loss_e = (1 - kd_alpha) * ce + kd_alpha * kd
            parts[f"kd{e}"] = kd.item()
        else:
            loss_e = ce
        parts[f"ce{e}"] = ce.item()
        total = total + w * loss_e
    return total, parts


@torch.no_grad()
def predict(model: ExitCNN, split: dict, cfg: AttrDict, device: torch.device | None = None) -> Tensor:
    """Logits for every exit on a split: [E, N, C] on CPU."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks: list[list[Tensor]] = [[] for _ in range(model.num_exits)]
    for xb, mb, _ in core.make_loader(split, batch_size=cfg.train.batch_size):
        outs = model(xb.to(device), mb.to(device))
        for e, z in enumerate(outs):
            chunks[e].append(z.float().cpu())
    model.train(was_training)
    return core.stack_exit_logits([torch.cat(c) for c in chunks])


def evaluate(model: ExitCNN, split: dict, cfg: AttrDict, device: torch.device | None = None) -> list[dict[str, float]]:
    """Per-exit metrics on a split (shallow exit first)."""
    logits = predict(model, split, cfg, device)
    return [core.compute_metrics(logits[e], split["y"], cfg.data.num_classes) for e in range(logits.shape[0])]


# ---------- training ----------

def fit(
    model: ExitCNN,
    train: dict,
    val: dict,
    cfg: AttrDict,
    *,
    exit_weights: list[float] | None = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 1.0,
    epochs: int | None = None,
    lr: float | None = None,
    log=print,
) -> list[dict]:
    """Train in place; the model ends up holding the best validation epoch. Returns history."""
    tr = cfg.train
    device = get_device(cfg)
    model.to(device)
    epochs = tr.epochs if epochs is None else epochs
    lr = tr.lr if lr is None else lr
    exit_weights = list(exit_weights) if exit_weights is not None else [1.0] * model.num_exits
    ce_weight = class_weights(train["y"], cfg.data.num_classes, tr.class_weight)
    if ce_weight is not None:
        ce_weight = ce_weight.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tr.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loader = core.make_loader(train, batch_size=tr.batch_size, shuffle=True)

    best_metric, best_state, best_epoch, bad_epochs = -float("inf"), None, -1, 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            outs = model(xb, mb)
            loss, _ = exit_loss(outs, yb, ce_weight, exit_weights, kd_alpha, kd_temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()

        val_metrics = evaluate(model, val, cfg, device)
        score = val_metrics[-1][tr.select_metric]
        rec = {"epoch": epoch, "train_loss": running / max(1, n_batches), "val": val_metrics, "time_s": time.time() - t0}
        history.append(rec)
        per_exit = "  ".join(f"e{e}:{m['bal_acc']:.3f}/{m['f1']:.3f}" for e, m in enumerate(val_metrics))
        log(f"epoch {epoch:3d}  loss {rec['train_loss']:.4f}  val bal_acc/f1  {per_exit}  ({rec['time_s']:.1f}s)")

        if score > best_metric:
            best_metric, best_epoch, bad_epochs = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if tr.patience and bad_epochs >= tr.patience:
                log(f"early stopping: no improvement in {tr.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"best epoch {best_epoch}: val {tr.select_metric} (final exit) = {best_metric:.4f}")
    model.eval()
    return history


# ---------- export ----------

def export(
    model: ExitCNN,
    name: str,
    variant: str,
    cfg: AttrDict,
    splits: dict[str, dict],
    *,
    kind: str,
    arch: dict,
    pruned: bool = False,
    log=print,
) -> dict:
    """Write everything downstream modules consume. `splits` holds at least 'val' and 'test'."""
    model = model.cpu().eval()
    out_dir = core.resolve_path(cfg, "outputs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(out_dir / "model.pt", model, kind=kind, arch=arch, variant=variant, pruned=pruned)

    probe = splits["val"]
    prof = core.profile(model, probe["X"][:1], probe["mask"][:1])
    core.save_flops(out_dir / "flops.json", prof)
    log(f"flops per exit: {prof['flops_per_exit']}  params={prof['params']}  size={prof['size_mb']:.3f} MB")

    results_csv = core.resolve_path(cfg, "results_csv")
    summary: dict[str, list[dict]] = {}
    for split_name in ("val", "test"):
        split = splits[split_name]
        logits = predict(model, split, cfg, torch.device("cpu"))
        core.save_logits(out_dir / f"logits_{split_name}.pt", logits, split["y"])
        summary[split_name] = []
        for e in range(model.num_exits):
            m = core.compute_metrics(logits[e], split["y"], cfg.data.num_classes)
            summary[split_name].append(m)
            row_variant = variant if model.num_exits == 1 else f"{variant}/exit{e}"
            core.append_result(
                results_csv,
                core.ResultRow(
                    model=name, variant=row_variant, budget="static", split=split_name,
                    flops_mean=prof["flops_per_exit"][e], size_mb=prof["size_mb"], **m,
                ),
            )
            log(f"{split_name:4s} exit{e}: acc {m['acc']:.3f}  bal_acc {m['bal_acc']:.3f}  f1 {m['f1']:.3f}")
    log(f"wrote {out_dir}/{{model.pt,flops.json,logits_val.pt,logits_test.pt}} and rows to {results_csv}")
    return {"flops": prof, "metrics": summary, "out_dir": str(out_dir)}


def load_splits(cfg: AttrDict, names: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict]:
    proc = core.resolve_path(cfg, "processed")
    return {s: core.load_split(proc / f"{s}.pt", num_classes=cfg.data.num_classes) for s in names}
