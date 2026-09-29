"""Tests for distance calculations and combination."""

from __future__ import annotations

from phailogeny.distance.blocks import (
    capability_distance,
    interface_distance,
    lineage_distance,
    prompt_distance,
    purpose_distance,
)
from phailogeny.distance.combine import combine_distances


def test_distance_functions_behave_expectedly() -> None:
    """The core block distances should return sensible values for similar and different inputs."""
    assert purpose_distance("review code", "review code") == 0.0
    assert prompt_distance("write code", "review code") < 1.0
    assert capability_distance({"read_file", "grep"}, {"read_file", "grep"}) == 0.0
    assert interface_distance({"text"}, {"json"}) > 0.0
    assert lineage_distance("alpha beta", "alpha beta") == 0.0


def test_combine_distances_renormalises_weights() -> None:
    """Missing distances should be ignored and remaining weights normalised."""
    score = combine_distances({"purpose": 0.2, "prompt": 0.4}, weights={"purpose": 1.0, "prompt": 3.0})
    assert abs(score - 0.35) < 1e-9
