import torch

from core.policy import confidence_scores, apply_threshold_policy


def test_max_softmax_confidence_shape_and_range():
    logits = torch.tensor(
        [
            [[4.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [3.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    conf = confidence_scores(logits, method="max_softmax")

    assert conf.shape == (3, 2)
    assert torch.all(conf >= 0.0)
    assert torch.all(conf <= 1.0)


def test_policy_uses_earliest_exit_that_meets_threshold():
    logits = torch.tensor(
        [
            [
                [4.0, 1.0],   # sample 1: confident immediately
                [1.1, 1.0],   # sample 2: not confident yet
                [1.0, 1.0],   # sample 3: not confident yet
            ],
            [
                [3.0, 1.0],
                [3.0, 1.0],   # sample 2 becomes confident here
                [1.2, 1.0],
            ],
            [
                [2.0, 1.0],
                [2.0, 1.0],
                [4.0, 1.0],   # sample 3 becomes confident here
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    thresholds = [0.90, 0.80, 0.70, 0.0]

    preds, exits, conf = apply_threshold_policy(
        logits,
        thresholds,
        method="max_softmax",
    )

    assert conf.shape == (4, 3)
    assert exits.tolist() == [0, 1, 2]
    assert preds.shape == (3,)


def test_final_exit_is_fallback():
    logits = torch.zeros((4, 5, 2), dtype=torch.float32)

    thresholds = [1.0, 1.0, 1.0, 0.0]

    preds, exits, _ = apply_threshold_policy(
        logits,
        thresholds,
        method="max_softmax",
    )

    assert exits.tolist() == [3, 3, 3, 3, 3]
    assert preds.shape == (5,)


def test_entropy_confidence_shape():
    logits = torch.tensor(
        [
            [[4.0, 1.0], [1.0, 1.0]],
            [[3.0, 1.0], [2.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    conf = confidence_scores(logits, method="entropy")

    assert conf.shape == (2, 2)
    assert torch.isfinite(conf).all()