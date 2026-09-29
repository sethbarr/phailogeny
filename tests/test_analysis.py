"""Tests for analysis helpers in the validation and report stack."""

from __future__ import annotations

from phailogeny.analysis.copy_debt import copy_debt
from phailogeny.analysis.drift import drift_pairs
from phailogeny.analysis.diversity import hill_numbers, redundancy_index
from phailogeny.analysis.merge import merge_candidates
from phailogeny.analysis.pangenome import pan_genome
from phailogeny.analysis.shared_needs import shared_needs


def test_analysis_helpers_are_stable() -> None:
    """The analysis helpers should return sensible summary records for a compact estate."""
    similarity = {("a", "b"): 0.9, ("b", "a"): 0.9}
    candidates = merge_candidates([("a", "b")], similarity, threshold=0.7)
    assert candidates[0]["status"] == "merge-candidate"

    families = {"shared": ["a", "b", "c"], "rare": ["d"]}
    clusters = {"cluster-1": ["a", "b"], "cluster-2": ["c", "d"]}
    assert shared_needs(families, clusters)[0][0] == "shared"

    pangenome = pan_genome({"a": ["shared"], "b": ["shared"], "c": ["rare"]})
    assert pangenome["a"] == ["shared"]

    assert copy_debt({"shared": ["a", "b", "c"]}) == [("shared", ["a", "b", "c"])]

    drift = drift_pairs([("a", "b")], {("a", "b"): 0.8})
    assert drift[0][2] == 0.8

    diversity = hill_numbers({"cluster-1": ["a", "b"], "cluster-2": ["c"]}, 1)
    assert diversity > 0
    assert redundancy_index({"cluster-1": ["a", "b"], "cluster-2": ["c"]}) == 1.5
