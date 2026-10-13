from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


def select_calibration_indices(
    y: Tensor,
    max_samples: int = 256,
    min_action_samples: int = 32,
    seed: int = 0,
) -> Tensor:
    """
    Select representative calibration examples.

    The project specifically requires calibration data to contain
    action windows (y == 1), rather than using almost entirely
    no-action windows.
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    y = y.detach().cpu()

    n = len(y)
    if n == 0:
        raise ValueError("Cannot calibrate on an empty dataset")

    positive = torch.nonzero(y == 1, as_tuple=False).flatten()
    negative = torch.nonzero(y == 0, as_tuple=False).flatten()

    # If the entire dataset fits, use it all.
    if n <= max_samples:
        return torch.arange(n)

    generator = torch.Generator().manual_seed(seed)

    # We deliberately reserve a useful part of calibration for
    # the rare positive/action class.
    desired_positive = max(min_action_samples, max_samples // 4)
    n_positive = min(len(positive), desired_positive, max_samples)
    n_negative = min(len(negative), max_samples - n_positive)

    # Fill any remaining capacity if one class did not have enough.
    remaining = max_samples - n_positive - n_negative

    if remaining > 0:
        extra_positive = min(
            len(positive) - n_positive,
            remaining,
        )
        n_positive += extra_positive
        remaining -= extra_positive

    if remaining > 0:
        extra_negative = min(
            len(negative) - n_negative,
            remaining,
        )
        n_negative += extra_negative

    if n_positive > 0:
        positive = positive[
            torch.randperm(len(positive), generator=generator)[:n_positive]
        ]

    if n_negative > 0:
        negative = negative[
            torch.randperm(len(negative), generator=generator)[:n_negative]
        ]

    indices = torch.cat([positive, negative])

    # Shuffle the final calibration order.
    order = torch.randperm(len(indices), generator=generator)
    return indices[order]


def prepare_ptq_fx(
    model: nn.Module,
    example_x: Tensor,
    example_mask: Tensor,
    backend: str = "onednn",
) -> nn.Module:
    """
    Prepare a model for FX graph-mode post-training static INT8
    quantization.

    The saved project data keeps mask as bool, but observers cannot
    calibrate bool tensors. Inside M6 only, the mask is converted to
    float 0/1 values. This preserves its masking semantics.
    """
    if example_x.shape[0] < 1:
        raise ValueError("example_x must contain at least one sample")

    if example_mask.shape[0] < 1:
        raise ValueError("example_mask must contain at least one sample")

    torch.backends.quantized.engine = backend

    model_fp32 = copy.deepcopy(model).cpu().eval()

    x = example_x[:1].detach().cpu().float()
    mask = example_mask[:1].detach().cpu().to(dtype=x.dtype)

    qconfig_mapping = get_default_qconfig_mapping(backend)

    prepared = prepare_fx(
        model_fp32,
        qconfig_mapping,
        (x, mask),
    )

    return prepared


def calibrate(
    prepared_model: nn.Module,
    x: Tensor,
    mask: Tensor,
    batch_size: int = 32,
) -> None:
    """Run representative data through observers before conversion."""
    if len(x) != len(mask):
        raise ValueError("x and mask must contain the same number of samples")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prepared_model.eval()

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].detach().cpu().float()

            # M1 data contract remains bool; only the quantized graph
            # receives the equivalent float 0/1 mask.
            mb = (
                mask[start : start + batch_size]
                .detach()
                .cpu()
                .to(dtype=xb.dtype)
            )

            prepared_model(xb, mb)


def quantize_ptq_fx(
    model: nn.Module,
    calibration_x: Tensor,
    calibration_mask: Tensor,
    backend: str = "onednn",
    batch_size: int = 32,
) -> nn.Module:
    """
    Perform complete post-training static INT8 quantization:
    prepare -> calibrate -> convert.
    """
    if len(calibration_x) == 0:
        raise ValueError("Calibration data must not be empty")

    prepared = prepare_ptq_fx(
        model,
        calibration_x[:1],
        calibration_mask[:1],
        backend=backend,
    )

    calibrate(
        prepared,
        calibration_x,
        calibration_mask,
        batch_size=batch_size,
    )

    quantized = convert_fx(prepared)
    quantized.eval()

    return quantized


def quantized_module_names(model: nn.Module) -> list[str]:
    """Return the names of modules that were actually quantized."""
    names: list[str] = []

    for name, module in model.named_modules():
        module_path = type(module).__module__.lower()

        if "quantized" in module_path:
            names.append(name)

    return names


def is_actually_quantized(model: nn.Module) -> bool:
    """True when FX conversion produced quantized modules."""
    return len(quantized_module_names(model)) > 0