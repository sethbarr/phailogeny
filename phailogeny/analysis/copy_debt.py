"""Identify copied modules that appear in multiple agents without reference tracking."""

from __future__ import annotations


def copy_debt(families: dict[str, list[str]], min_agents: int = 3) -> list[tuple[str, list[str]]]:
    """Return copied families with at least the minimum number of distinct agents."""
    return [
        (family, members)
        for family, members in families.items()
        if len(members) >= min_agents
    ]
