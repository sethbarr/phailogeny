"""Drift detection for homologous but functionally divergent agents."""

from __future__ import annotations


def drift_pairs(
    pairs: list[tuple[str, str]],
    functional_distance: dict[tuple[str, str], float],
    threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Return pairs whose functional distance exceeds the threshold."""
    results: list[tuple[str, str, float]] = []
    for left, right in pairs:
        score = functional_distance.get((left, right), functional_distance.get((right, left), 0.0))
        if score >= threshold:
            results.append((left, right, score))
    return results
