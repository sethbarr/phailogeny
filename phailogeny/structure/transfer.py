"""Transfer-detection heuristics for copied or reused modules."""

from __future__ import annotations


def detect_transfer_candidates(
    agent_pairs: list[tuple[str, str]],
    module_families: dict[str, list[str]],
    vertical_distance: dict[tuple[str, str], float] | None = None,
    functional_distance: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, object]]:
    """Return a record for each candidate transfer relation."""
    results: list[dict[str, object]] = []
    for left_agent, right_agent in agent_pairs:
        pair = (left_agent, right_agent)
        family_matches = [
            family
            for family, members in module_families.items()
            if left_agent in members and right_agent in members
        ]

        if not family_matches:
            continue

        if vertical_distance is not None and functional_distance is not None:
            vertical = vertical_distance.get(pair, 1.0)
            functional = functional_distance.get(pair, 0.0)
            if vertical >= 0.3 or functional <= 0.5:
                continue

        results.append({"pair": pair, "families": family_matches, "status": "candidate-transfer"})
    return results
