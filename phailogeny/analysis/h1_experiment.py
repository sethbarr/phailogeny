"""Synthetic H1 experiment scaffolding for M0 vs M1 lineage models."""

from __future__ import annotations


def h1_summary(results: list[dict[str, float]]) -> dict[str, float]:
    """Compute a simple summary of H1 experimental outcomes.

    Args:
        results: A list of dictionaries containing at least a 'm0' and 'm1' score.

    Returns:
        A dictionary of summary values for the experiment.
    """
    if not results:
        return {"mean_m0": 0.0, "mean_m1": 0.0, "mean_delta": 0.0}

    mean_m0 = sum(item.get("m0", 0.0) for item in results) / len(results)
    mean_m1 = sum(item.get("m1", 0.0) for item in results) / len(results)
    return {
        "mean_m0": mean_m0,
        "mean_m1": mean_m1,
        "mean_delta": mean_m1 - mean_m0,
    }
