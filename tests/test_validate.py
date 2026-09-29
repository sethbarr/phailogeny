"""Tests for validation metrics and synthetic generator scaffolding."""

from __future__ import annotations

from phailogeny.analysis.validate import adjusted_rand_index, validation_summary
from phailogeny.synth.generator import SyntheticEstateGenerator


def test_validation_metrics_return_expected_values() -> None:
    """The validation metrics should return a sensible accuracy score for aligned labels."""
    assert adjusted_rand_index(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    summary = validation_summary(["a", "b", "a"], ["a", "b", "a"])
    assert summary["accuracy"] == 1.0


def test_synthetic_generator_creates_agents_and_truth() -> None:
    """The synthetic estate generator should produce agents and a stored truth snapshot."""
    generator = SyntheticEstateGenerator(roles=2, agents_per_role=2, seed=7)
    agents = generator.generate()
    assert len(agents) == 4
    assert agents[0].role in {"role-0", "role-1"}
