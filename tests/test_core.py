import math

import pytest
import torch

import core


@pytest.fixture(scope="module")
def cfg():
    return core.load_config()


@pytest.fixture
def split(cfg):
    return core.make_mock_split(n=32, T=core.window_frames(cfg), P=cfg.data.max_people, match_ids=["M1", "M3"], seed=1)


def test_config_loads(cfg):
    assert cfg.data.num_classes == 2
    assert core.window_frames(cfg) == round(cfg.data.fps * cfg.data.window_seconds)
    assert set(cfg.data.splits) == {"train", "val", "test"}


def test_mock_split_matches_contract(split, cfg):
    core.validate_split(split, num_classes=cfg.data.num_classes)
    assert split["X"].dtype == torch.float32 and split["X"].ndim == 4
    assert split["mask"].dtype == torch.bool
    assert split["y"].dtype == torch.int64
    assert (split["X"][~split["mask"]] == 0).all()


def test_validate_rejects_bad_dtype(split):
    bad = dict(split, y=split["y"].float())
    with pytest.raises(ValueError):
        core.validate_split(bad)


def test_save_and_load_roundtrip(split, tmp_path):
    p = tmp_path / "val.pt"
    core.save_split(split, p, num_classes=2)
    loaded = core.load_split(p, num_classes=2)
    assert torch.equal(loaded["X"], split["X"])
    assert loaded["meta"]["match_id"] == split["meta"]["match_id"]


def test_write_mock_dataset(cfg, tmp_path):
    paths = core.write_mock_dataset(cfg, out_dir=tmp_path)
    assert set(paths) == set(core.SPLITS)
    d = core.load_split(paths["train"], num_classes=cfg.data.num_classes)
    assert d["X"].shape[0] == cfg.mock.n_train
    assert set(d["meta"]["match_id"]) <= set(cfg.data.splits["train"])


def test_dummy_model_interface(split, cfg):
    model = core.DummyExitModel(max_people=cfg.data.max_people, num_exits=3).eval()
    outs = model(split["X"], split["mask"])
    assert model.num_exits == 3 and len(outs) == 3
    for o in outs:
        assert tuple(o.shape) == (32, cfg.data.num_classes)
    # forward_until must give exactly the same logits as the full forward at that exit
    for e in range(model.num_exits):
        assert torch.allclose(model.forward_until(split["X"], split["mask"], e), outs[e])
    assert tuple(core.stack_exit_logits(outs).shape) == (3, 32, cfg.data.num_classes)
    with pytest.raises(IndexError):
        model.forward_until(split["X"], split["mask"], 3)


def test_flops_increase_with_depth(split, cfg):
    model = core.DummyExitModel(max_people=cfg.data.max_people, num_exits=4)
    prof = core.profile(model, split["X"], split["mask"])
    f = prof["flops_per_exit"]
    assert len(f) == 4 and all(f[i] < f[i + 1] for i in range(3))
    assert prof["params"] == sum(p.numel() for p in model.parameters())
    assert prof["size_mb"] > 0


def test_metrics_known_values():
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    preds = torch.tensor([0, 0, 1, 1, 1, 0])
    logits = torch.nn.functional.one_hot(preds, 2).float()
    m = core.compute_metrics(logits, y)
    assert math.isclose(m["acc"], 4 / 6)
    assert math.isclose(m["bal_acc"], 4 / 6)
    assert math.isclose(m["f1"], 2 / 3)     # tp=2, fp=1, fn=1


def test_metrics_perfect_and_all_negative():
    y = torch.tensor([0, 0, 1, 1])
    perfect = torch.nn.functional.one_hot(y, 2).float()
    assert core.compute_metrics(perfect, y) == {"acc": 1.0, "bal_acc": 1.0, "f1": 1.0}
    all_neg = torch.nn.functional.one_hot(torch.zeros_like(y), 2).float()
    m = core.compute_metrics(all_neg, y)
    assert m["acc"] == 0.5 and m["bal_acc"] == 0.5 and m["f1"] == 0.0


def test_artifacts_roundtrip(tmp_path):
    logits = torch.randn(3, 10, 2)
    y = torch.randint(0, 2, (10,))
    core.save_logits(tmp_path / "logits_val.pt", logits, y)
    l2, y2 = core.load_logits(tmp_path / "logits_val.pt")
    assert torch.equal(l2, logits) and torch.equal(y2, y)

    core.save_flops(tmp_path / "flops.json", {"tool": "x", "flops_per_exit": [1, 2, 3], "params": 5, "size_mb": 0.1})
    assert core.load_flops(tmp_path / "flops.json")["flops_per_exit"] == [1, 2, 3]
    with pytest.raises(ValueError):
        core.save_flops(tmp_path / "bad.json", {"flops_per_exit": [1]})

    core.save_thresholds(tmp_path / "thresholds.json", {100: [0.9, 0.8, 0.0], 20: [0.5, 0.5, 0.0]})
    assert core.load_thresholds(tmp_path / "thresholds.json") == {"100": [0.9, 0.8, 0.0], "20": [0.5, 0.5, 0.0]}


def test_results_csv(tmp_path):
    p = tmp_path / "results.csv"
    row = core.ResultRow("dummy", "fp32", "static", "val", 1000.0, 0.5, 0.5, 0.3, 0.01)
    core.append_result(p, row)
    core.append_result(p, row)
    df = core.load_results(p)
    assert len(df) == 2 and list(df.columns) == core.RESULT_COLUMNS
    assert math.isnan(df.latency_ms.iloc[0])
