"""Merge-candidate analysis for convergent or redundant agents."""

from __future__ import annotations


def merge_candidates(
    pairs: list[tuple[str, str]],
    similarity: dict[tuple[str, str], float],
    threshold: float = 0.7,
) -> list[dict[str, object]]:
    """Return candidate merge pairs whose similarity exceeds the threshold."""
    results: list[dict[str, object]] = []
    for left, right in pairs:
        score = similarity.get((left, right), similarity.get((right, left), 0.0))
        if score >= threshold:
            results.append({"pair": (left, right), "similarity": score, "status": "merge-candidate"})
    return results
