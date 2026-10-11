from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor


ConfidenceMode = Literal["max_softmax", "entropy"]


def _validate_logits(logits: Tensor) -> None:
    """
    Validate stacked early-exit logits.

    Expected shape:
        [E, N, C]

    E = number of exits
    N = number of samples/windows
    C = number of classes
    """
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [E, N, C], got {tuple(logits.shape)}"
        )

    if logits.shape[0] < 1:
        raise ValueError("logits must contain at least one exit")

    if logits.shape[2] < 2:
        raise ValueError("logits must contain at least two classes")


def confidence_scores(
    logits: Tensor,
    *,
    method: ConfidenceMode = "max_softmax",
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute a confidence score for every exit and sample.

    Input:
        logits: [E, N, C]

    Output:
        confidence: [E, N]

    Higher scores always mean more confidence.

    Methods
    -------
    max_softmax:
        max_c softmax(logits)[c]

    entropy:
        1 - normalized entropy.

        Raw entropy is uncertainty, so it is inverted here such that
        larger values again mean more confidence.

        confidence =
            1 - H(p) / log(C)
    """
    _validate_logits(logits)

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)

    if method == "max_softmax":
        return probs.max(dim=-1).values

    if method == "entropy":
        eps = torch.finfo(probs.dtype).eps
        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)

        num_classes = logits.shape[-1]

        if num_classes <= 1:
            return torch.ones_like(entropy)

        normalized_entropy = entropy / math.log(num_classes)

        # Convert uncertainty to confidence:
        # 1 = very confident
        # 0 = maximally uncertain
        return 1.0 - normalized_entropy

    raise ValueError(
        f"Unknown confidence method {method!r}. "
        "Expected 'max_softmax' or 'entropy'."
    )


def apply_threshold_policy(
    logits: Tensor,
    thresholds: list[float] | tuple[float, ...] | Tensor,
    *,
    method: ConfidenceMode = "max_softmax",
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Apply an early-exit threshold policy to stacked logits.

    Parameters
    ----------
    logits:
        Tensor [E, N, C].

    thresholds:
        One threshold per exit.

        For exits 0 ... E-2, a sample exits when:

            confidence >= threshold

        The final exit is always used as the fallback if no previous
        exit satisfied its threshold.

    method:
        'max_softmax' or 'entropy'.

    temperature:
        Optional softmax temperature.

    Returns
    -------
    predictions:
        [N] predicted class.

    exit_ids:
        [N] zero-based exit index used for each sample.

    confidence:
        [E, N] confidence score for every exit/sample.
    """
    _validate_logits(logits)

    num_exits, num_samples, _ = logits.shape

    thresholds_tensor = torch.as_tensor(
        thresholds,
        dtype=logits.dtype,
        device=logits.device,
    )

    if thresholds_tensor.ndim != 1:
        raise ValueError("thresholds must be a 1-D sequence")

    if len(thresholds_tensor) != num_exits:
        raise ValueError(
            f"Expected {num_exits} thresholds, "
            f"got {len(thresholds_tensor)}"
        )

    if not torch.all(
        (thresholds_tensor >= 0.0) & (thresholds_tensor <= 1.0)
    ):
        raise ValueError("all thresholds must be between 0 and 1")

    confidence = confidence_scores(
        logits,
        method=method,
        temperature=temperature,
    )

    # Default: every sample reaches the final exit.
    exit_ids = torch.full(
        (num_samples,),
        num_exits - 1,
        dtype=torch.long,
        device=logits.device,
    )

    unresolved = torch.ones(
        num_samples,
        dtype=torch.bool,
        device=logits.device,
    )

    # Try exits from shallowest to deepest.
    # The final exit is always the fallback.
    for exit_idx in range(num_exits - 1):
        take_exit = (
            unresolved
            & (confidence[exit_idx] >= thresholds_tensor[exit_idx])
        )

        exit_ids[take_exit] = exit_idx
        unresolved[take_exit] = False

    sample_ids = torch.arange(num_samples, device=logits.device)

    selected_logits = logits[
        exit_ids,
        sample_ids,
        :,
    ]

    predictions = selected_logits.argmax(dim=-1)

    return predictions, exit_ids, confidence


def run_policy_single(
    exit_logits: Tensor,
    thresholds: list[float] | tuple[float, ...] | Tensor,
    *,
    method: ConfidenceMode = "max_softmax",
    temperature: float = 1.0,
) -> tuple[int, int, list[float]]:
    """
    Apply the policy to one streaming window.

    Parameters
    ----------
    exit_logits:
        [E, C] logits for one input window.

    Returns
    -------
    prediction:
        Predicted class.

    exit_id:
        Zero-based exit used.

    confidences:
        Confidence at every exit.
    """
    if exit_logits.ndim != 2:
        raise ValueError(
            f"exit_logits must have shape [E, C], "
            f"got {tuple(exit_logits.shape)}"
        )

    stacked = exit_logits.unsqueeze(1)  # [E, 1, C]

    predictions, exit_ids, confidence = apply_threshold_policy(
        stacked,
        thresholds,
        method=method,
        temperature=temperature,
    )

    return (
        int(predictions[0].item()),
        int(exit_ids[0].item()),
        [float(x) for x in confidence[:, 0].detach().cpu()],
    )


def exit_histogram(exit_ids: Tensor, num_exits: int) -> list[int]:
    """
    Count how many samples used each exit.
    """
    if exit_ids.ndim != 1:
        raise ValueError("exit_ids must be a 1-D tensor")

    counts = torch.bincount(
        exit_ids.long().cpu(),
        minlength=num_exits,
    )

    return [int(x) for x in counts.tolist()]


def mean_flops(
    exit_ids: Tensor,
    flops_per_exit: list[int] | tuple[int, ...] | Tensor,
) -> float:
    """
    Compute mean FLOPs per sample for a chosen set of exits.

    flops_per_exit must contain cumulative FLOPs required to reach
    each exit, exactly as produced by M4.
    """
    flops = torch.as_tensor(
        flops_per_exit,
        dtype=torch.float64,
    )

    if flops.ndim != 1:
        raise ValueError("flops_per_exit must be a 1-D sequence")

    if exit_ids.numel() == 0:
        return 0.0

    if int(exit_ids.max()) >= len(flops):
        raise ValueError(
            "exit_ids contains an exit that is missing from "
            "flops_per_exit"
        )

    selected = flops[exit_ids.long().cpu()]

    return float(selected.mean().item())


def budget_thresholds(
    thresholds_by_budget: dict[str, list[float]],
    budget: str | int,
) -> list[float]:
    """
    Retrieve thresholds for a requested budget level.

    Example thresholds file:

        {
            "100": [0.99, 0.99, 0.99, 0.0],
            "50":  [0.85, 0.80, 0.75, 0.0],
            "20":  [0.65, 0.60, 0.55, 0.0]
        }
    """
    key = str(budget)

    if key not in thresholds_by_budget:
        available = ", ".join(sorted(thresholds_by_budget))
        raise KeyError(
            f"Unknown budget {key!r}. Available budgets: {available}"
        )

    return [float(x) for x in thresholds_by_budget[key]]