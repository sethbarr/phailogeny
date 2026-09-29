"""Pan-genome summaries for agent-family presence across the estate."""

from __future__ import annotations


def pan_genome(agent_families: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return a simple family-presence map with per-agent family membership."""
    return {agent: sorted(families) for agent, families in agent_families.items()}
