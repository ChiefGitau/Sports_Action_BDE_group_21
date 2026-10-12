import pytest
import torch

import core
import models


@pytest.fixture(scope="module")
def cfg():
    return core.load_config()


@pytest.fixture(scope="module")
def splits(cfg):
    T, P = core.window_frames(cfg), cfg.data.max_people
    return {
        "train": core.make_mock_split(n=96, T=T, P=P, match_ids=["M1"], seed=11),
        "val": core.make_mock_split(n=32, T=T, P=P, match_ids=["M2"], seed=12),
        "test": core.make_mock_split(n=32, T=T, P=P, match_ids=["M7"], seed=13),
    }


@pytest.mark.parametrize("kind", ["static", "early_exit"])
def test_models_follow_exit_interface(cfg, splits, kind):
    d = splits["val"]
    m = models.build_model(cfg, kind).eval()
    outs = m(d["X"], d["mask"])
    assert len(outs) == m.num_exits
    assert m.num_exits == (1 if kind == "static" else len(cfg.model.exit_after))
    for e, o in enumerate(outs):
        assert tuple(o.shape) == (32, cfg.data.num_classes)
        assert torch.allclose(m.forward_until(d["X"], d["mask"], e), o, atol=1e-6)
    with pytest.raises(IndexError):
        m.forward_until(d["X"], d["mask"], m.num_exits)


def test_static_and_early_exit_share_trunk_cost(cfg, splits):
    d = splits["val"]
    s = models.build_model(cfg, "static")
    ee = models.build_model(cfg, "early_exit")
    fs = core.profile(s, d["X"], d["mask"])["flops_per_exit"]
    fe = core.profile(ee, d["X"], d["mask"])["flops_per_exit"]
    assert all(fe[i] < fe[i + 1] for i in range(len(fe) - 1))
    # The last exit costs the static baseline plus the earlier heads the device evaluated
    # and rejected on the way (one Linear(channels, C) each: 2*in*out FLOPs at batch 1).
    head_flops = 2 * cfg.model.channels * cfg.data.num_classes
    assert fe[-1] == fs[0] + (ee.num_exits - 1) * head_flops
    assert fe[0] < fs[0]


def test_exit_after_validation(cfg):
    with pytest.raises(ValueError):
        models.build_model(cfg, "early_exit", exit_after=[1, 2])   # must end at num_blocks
    with pytest.raises(ValueError):
        models.build_model(cfg, "early_exit", exit_after=[4])      # needs >= 2 exits


def test_exit_loss_kd_only_on_early_exits():
    y = torch.tensor([0, 1, 1, 0])
    outs = [torch.randn(4, 2, requires_grad=True) for _ in range(3)]
    loss, parts = models.exit_loss(outs, y, None, [1.0, 1.0, 1.0], kd_alpha=0.5, kd_temperature=2.0)
    loss.backward()
    assert {"ce0", "ce1", "ce2", "kd0", "kd1"} == set(parts)
    assert all(o.grad is not None for o in outs)
    with pytest.raises(ValueError):
        models.exit_loss(outs, y, None, [1.0], 0.0, 1.0)


def test_fit_runs_and_selects_best(cfg, splits):
    cfg_fast = core.AttrDict.from_dict(dict(cfg))
    cfg_fast.train = core.AttrDict.from_dict(dict(cfg.train, device="cpu", patience=0))
    m = models.build_model(cfg, "early_exit")
    hist = models.fit(m, splits["train"], splits["val"], cfg_fast,
                      exit_weights=cfg.early_exit.exit_loss_weights, kd_alpha=0.5, kd_temperature=2.0,
                      epochs=2, log=lambda *_: None)
    assert len(hist) == 2 and len(hist[0]["val"]) == m.num_exits
    assert not m.training
    logits = models.predict(m, splits["val"], cfg_fast)
    assert tuple(logits.shape) == (m.num_exits, 32, cfg.data.num_classes)


def test_prune_is_physical_and_reloads(cfg, splits, tmp_path):
    d = splits["val"]
    m = models.build_model(cfg, "static").eval()
    before = core.profile(m, d["X"], d["mask"])
    models.prune_channels(m, d["X"], d["mask"], 0.5)
    after = core.profile(m, d["X"], d["mask"])
    assert all(w == cfg.model.channels // 2 for w in m.channel_widths())
    assert after["flops_per_exit"][0] < 0.4 * before["flops_per_exit"][0]
    assert after["params"] < 0.4 * before["params"]
    assert tuple(m(d["X"], d["mask"])[0].shape) == (32, cfg.data.num_classes)

    models.save_checkpoint(tmp_path / "m.pt", m, "static", {}, "fp32", pruned=True)
    m2, meta = models.load_checkpoint(tmp_path / "m.pt")
    assert m2.channel_widths() == m.channel_widths() and meta["variant"] == "fp32"
    assert torch.allclose(m2(d["X"], d["mask"])[0], m(d["X"], d["mask"])[0])


def test_checkpoint_roundtrip_unpruned(cfg, splits, tmp_path):
    d = splits["val"]
    m = models.build_model(cfg, "early_exit").eval()
    arch = models.arch_from_config(cfg, "early_exit")
    models.save_checkpoint(tmp_path / "m.pt", m, "early_exit", arch, "fp32+kd")
    m2, meta = models.load_checkpoint(tmp_path / "m.pt")
    assert meta["kind"] == "early_exit" and m2.num_exits == m.num_exits
    for a, b in zip(m(d["X"], d["mask"]), m2(d["X"], d["mask"])):
        assert torch.allclose(a, b)


def test_export_writes_all_artifacts(cfg, splits, tmp_path):
    cfg_tmp = core.AttrDict.from_dict(dict(cfg))
    cfg_tmp.paths = core.AttrDict.from_dict(dict(cfg.paths, outputs=str(tmp_path), results_csv=str(tmp_path / "results.csv")))
    m = models.build_model(cfg, "early_exit").eval()
    out = models.export(m, "ee_test", "fp32+kd", cfg_tmp, splits,
                        kind="early_exit", arch=models.arch_from_config(cfg, "early_exit"), log=lambda *_: None)
    out_dir = tmp_path / "ee_test"
    for f in ("model.pt", "flops.json", "logits_val.pt", "logits_test.pt"):
        assert (out_dir / f).exists()
    logits, y = core.load_logits(out_dir / "logits_test.pt")
    assert tuple(logits.shape) == (m.num_exits, 32, cfg.data.num_classes) and y.shape == (32,)
    assert core.load_flops(out_dir / "flops.json")["flops_per_exit"] == out["flops"]["flops_per_exit"]
    df = core.load_results(tmp_path / "results.csv")
    assert len(df) == 2 * m.num_exits and set(df.split) == {"val", "test"}
    assert set(df.variant) == {f"fp32+kd/exit{e}" for e in range(m.num_exits)}
