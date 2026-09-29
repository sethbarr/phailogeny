"""Hill-number diversity and redundancy reporting."""

from __future__ import annotations

import math


def hill_numbers(cluster_membership: dict[str, list[str]], q: int) -> float:
    """Compute the Hill number of order q over a cluster-membership map."""
    if not cluster_membership:
        return 0.0

    total_agents = sum(len(members) for members in cluster_membership.values())
    if total_agents == 0:
        return 0.0

    if q == 0:
        return float(len(cluster_membership))

    if q == 1:
        proportions = [len(members) / total_agents for members in cluster_membership.values()]
        entropy = -sum(p * math.log(p) for p in proportions if p > 0)
        return math.exp(entropy)

    if q == 2:
        proportions = [len(members) / total_agents for members in cluster_membership.values()]
        return 1.0 / sum(p * p for p in proportions)

    raise ValueError("q must be one of 0, 1, or 2")


def redundancy_index(cluster_membership: dict[str, list[str]]) -> float:
    """Return agents per cluster as a basic redundancy metric."""
    if not cluster_membership:
        return 0.0
    total_agents = sum(len(members) for members in cluster_membership.values())
    return total_agents / len(cluster_membership)
