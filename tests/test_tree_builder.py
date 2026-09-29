"""Tests for the tree builder placeholder tied to the equal-weight matrix."""

from __future__ import annotations

from phailogeny.structure.tree_builder import nj_tree_from_matrix


def test_nj_tree_from_matrix_builds_a_newick_like_structure() -> None:
    """The tree builder should produce a stable structure from a matrix."""
    matrix = {
        "a": [0.0, 0.2],
        "b": [0.2, 0.0],
    }
    result = nj_tree_from_matrix(matrix)
    assert "a" in result["nodes"]
    assert "b" in result["nodes"]
    assert result["newick"].startswith("(")
