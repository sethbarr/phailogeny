"""Combine multiple block distances into a single estate-level distance."""

from __future__ import annotations


def combine_distances(
    distances: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine a block-distance dictionary with weights.

    Missing distances are ignored and the remaining weights are renormalised.
    """
    if not distances:
        return 0.0

    if weights is None:
        weights = {name: 1.0 for name in distances}

    valid = {name: value for name, value in distances.items() if value is not None}
    if not valid:
        return 0.0

    total_weight = sum(weights.get(name, 1.0) for name in valid)
    if total_weight == 0:
        return 0.0

    combined = sum(valid[name] * weights.get(name, 1.0) for name in valid)
    return combined / total_weight
