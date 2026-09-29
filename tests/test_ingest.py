"""Tests for corpus ingestion helpers."""

from __future__ import annotations

from phailogeny.ingest.claude_subagents import ClaudeSubagentIngester


def test_claude_subagent_ingester_parses_frontmatter() -> None:
    """The parser should read YAML frontmatter and set the prompt body from the markdown."""
    record = ClaudeSubagentIngester().parse_file("tests/fixtures/claude-subagent.md")

    assert record["name"] == "code-reviewer"
    assert record["purpose"] == "Reviews code for quality and correctness."
    assert record["framework"] == "claude-code"
    assert record["tools"] == ["read_file", "grep"]
    assert "You are a careful code reviewer." in record["prompt"]


def test_invalid_yaml_frontmatter_falls_back_to_key_value_lines(tmp_path) -> None:
    path = tmp_path / "agent.md"
    path.write_text("---\nname: ui\ndescription: Use this. Examples: user: hi\ntools: Read, Write\n---\nBody\n", encoding="utf-8")
    record = ClaudeSubagentIngester().parse_file(path)
    assert record["name"] == "ui" and record["purpose"].startswith("Use this.")
    assert record["tools"] == ["Read", "Write"]
