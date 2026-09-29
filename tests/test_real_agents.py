"""Tests for real-agent corpus analysis helpers."""

from __future__ import annotations

from phailogeny.analysis.real_agents import load_real_agent_records, summarize_real_agents


def test_real_agent_helpers_can_load_and_summarize_real_corpora() -> None:
    """The real-agent analysis helper should parse markdown agent definitions and summarize their similarity."""
    records = load_real_agent_records("tests/fixtures")
    assert len(records) >= 1

    summary = summarize_real_agents(records)
    assert summary["count"] >= 1
    assert "top_pairs" in summary
