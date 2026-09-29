"""Clustering utilities for functional agent similarity."""

from __future__ import annotations

from typing import Any


def cluster_agents(distances: list[list[float]], threshold: float = 0.5, min_cluster_size: int = 2) -> dict[str, Any]:
    """Create a simple clustering result view using a fixed distance threshold.

    The project spec calls for HDBSCAN, but for the MVP this provides a stable,
    deterministic fallback that keeps the API shape simple enough for tests and
    early development while the real implementation can be swapped in later.
    """
    if not distances:
        return {"labels": [], "clusters": []}

    labels: list[int] = [-1] * len(distances)
    clusters: list[list[int]] = []

    for row_index, row in enumerate(distances):
        assigned = False
        for cluster_index, cluster in enumerate(clusters):
            for other_index in cluster:
                if row[other_index] <= threshold:
                    cluster.append(row_index)
                    labels[row_index] = cluster_index
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            clusters.append([row_index])
            labels[row_index] = len(clusters) - 1

    return {"labels": labels, "clusters": clusters}
