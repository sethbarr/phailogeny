"""Validation metrics for synthetic estates and calibration experiments."""

from __future__ import annotations


def adjusted_rand_index(labels: list[str], truth: list[str]) -> float:
    """Compute a simple proxy for ARI on nominal cluster labels."""
    if len(labels) != len(truth):
        raise ValueError("labels and truth must have the same length")
    if not labels:
        return 0.0

    agreements = sum(1 for left, right in zip(labels, truth) if left == right)
    return agreements / len(labels)


def validation_summary(labels: list[str], truth: list[str]) -> dict[str, float]:
    """Return precision-like metrics for basic validation use."""
    if not labels:
        return {"accuracy": 0.0}
    accuracy = adjusted_rand_index(labels, truth)
    return {"accuracy": accuracy}
