"""Aggregate agent feature vectors into a matrix for downstream clustering and tree methods."""

from __future__ import annotations

from phailogeny.distance.blocks import capability_distance, interface_distance, lineage_distance, prompt_distance, purpose_distance


def build_agent_matrix(records: list[dict[str, object]]) -> dict[str, list[float]]:
    """Build a pairwise distance matrix from a list of normalized agent records.

    This is a lightweight real-corpus matrix that uses the equal-weight design chosen
    for the MVP, with each block contributing to the final pairwise comparison.
    """
    matrix: dict[str, list[float]] = {}
    for index, left in enumerate(records):
        left_id = str(left.get("name") or f"agent-{index}")
        row: list[float] = []
        for right in records:
            right_id = str(right.get("name") or f"agent-{index}")
            if left_id == right_id and left is right:
                row.append(0.0)
                continue

            purpose = purpose_distance(str(left.get("purpose") or ""), str(right.get("purpose") or ""))
            prompt = prompt_distance(str(left.get("prompt") or ""), str(right.get("prompt") or ""))
            capability = capability_distance(left.get("tools") or [], right.get("tools") or [])
            interface = interface_distance(left.get("input_modes") or [], right.get("input_modes") or [])
            lineage = lineage_distance(str(left.get("prompt") or ""), str(right.get("prompt") or ""))

            weights = {"purpose": 1.0, "prompt": 1.0, "capability": 1.0, "interface": 1.0, "lineage": 1.0}
            combined = (
                purpose * weights["purpose"]
                + prompt * weights["prompt"]
                + capability * weights["capability"]
                + interface * weights["interface"]
                + lineage * weights["lineage"]
            ) / sum(weights.values())
            row.append(combined)
        matrix[left_id] = row
    return matrix
