import torch

from compression.quantization import (
    is_actually_quantized,
    quantize_ptq_fx,
    select_calibration_indices,
)
from core.model import DummyExitModel


def _fake_data(n: int = 24):
    torch.manual_seed(0)

    X = torch.randn(n, 12, 23, 2)
    mask = torch.ones(n, 12, 23, dtype=torch.bool)

    y = torch.zeros(n, dtype=torch.int64)
    y[:6] = 1

    return X, mask, y


def test_calibration_selection_contains_action_windows():
    _, _, y = _fake_data()

    idx = select_calibration_indices(
        y,
        max_samples=8,
        min_action_samples=3,
        seed=0,
    )

    assert idx.shape == (8,)
    assert len(torch.unique(idx)) == 8

    selected_y = y[idx]

    # Calibration must deliberately contain positive/action windows.
    assert int((selected_y == 1).sum()) >= 3


def test_small_dataset_uses_all_calibration_samples():
    _, _, y = _fake_data(n=8)

    idx = select_calibration_indices(
        y,
        max_samples=32,
        min_action_samples=2,
        seed=0,
    )

    assert torch.equal(idx, torch.arange(8))


def test_fx_ptq_produces_real_quantized_modules():
    X, mask, _ = _fake_data()

    model = DummyExitModel(
        max_people=23,
        num_exits=4,
    ).eval()

    quantized = quantize_ptq_fx(
        model,
        X[:8],
        mask[:8],
        backend="onednn",
        batch_size=4,
    )

    assert is_actually_quantized(quantized)


def test_quantized_model_preserves_early_exit_output_shapes():
    X, mask, _ = _fake_data()

    model = DummyExitModel(
        max_people=23,
        num_exits=4,
    ).eval()

    quantized = quantize_ptq_fx(
        model,
        X[:8],
        mask[:8],
        backend="onednn",
        batch_size=4,
    )

    with torch.no_grad():
        outputs = quantized(
            X[:2],
            mask[:2].to(dtype=X.dtype),
        )

    assert len(outputs) == 4

    for output in outputs:
        assert tuple(output.shape) == (2, 2)