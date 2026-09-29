"""Tests for the equal-weight real-agent distance matrix builder."""

from __future__ import annotations

from phailogeny.analysis.agent_matrix import build_agent_matrix


def test_build_agent_matrix_makes_equal_weight_matrix() -> None:
    """The matrix builder should combine the major feature blocks with equal weights."""
    records = [
        {"name": "agent-a", "purpose": "review code", "prompt": "review code and fix issues", "tools": ["read_file"], "input_modes": ["text"]},
        {"name": "agent-b", "purpose": "review code", "prompt": "review code and fix issues", "tools": ["read_file"], "input_modes": ["text"]},
    ]
    matrix = build_agent_matrix(records)
    assert matrix["agent-a"][0] == 0.0
    assert matrix["agent-b"][1] == 0.0
