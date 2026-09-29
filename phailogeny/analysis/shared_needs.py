"""Shared-needs analysis across clusters and shell modules."""

from __future__ import annotations


def shared_needs(families: dict[str, list[str]], clusters: dict[str, list[str]]) -> list[tuple[str, int]]:
    """Return shell-family candidates ranked by the number of clusters they touch."""
    results: list[tuple[str, int]] = []
    for family, members in families.items():
        touched_clusters = set()
        for cluster_id, members_in_cluster in clusters.items():
            if any(member in members for member in members_in_cluster):
                touched_clusters.add(cluster_id)
        results.append((family, len(touched_clusters)))
    return sorted(results, key=lambda item: (-item[1], item[0]))
