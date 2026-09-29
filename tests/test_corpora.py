"""Tests for the additional public agent corpora ingesters."""

from __future__ import annotations

import json
from pathlib import Path

from phailogeny.ingest.a2a_registry import A2ARegistryIngester
from phailogeny.ingest.gptzoo import GPTZooIngester
from phailogeny.ingest.n8n import N8NIngester


def test_gptzoo_ingester_parses_metadata() -> None:
    """GPTZoo metadata should be loaded into the normalized profile shape."""
    path = Path("tests/fixtures/gptzoo.json")
    path.write_text(json.dumps({"name": "Example GPT", "description": "Helps with writing", "tools": ["web", "code"]}), encoding="utf-8")
    record = GPTZooIngester().parse_file(path)
    assert record["framework"] == "gpt"
    assert record["tools"] == ["web", "code"]


def test_a2a_registry_ingester_parses_cards() -> None:
    """A2A cards should map cleanly to the agent schema."""
    path = Path("tests/fixtures/a2a.json")
    path.write_text(json.dumps({"name": "Support Agent", "description": "Handles support", "skills": [{"name": "triage"}]}), encoding="utf-8")
    record = A2ARegistryIngester().parse_file(path)
    assert record["framework"] == "a2a"
    assert record["tools"] == ["triage"]


def test_n8n_ingester_parses_workflow_json() -> None:
    """n8n workflow JSON should produce a normalized workflow record."""
    path = Path("tests/fixtures/n8n.json")
    path.write_text(json.dumps({"name": "Workflow A", "description": "Runs tasks", "nodes": [{"type": "webhook"}, {"type": "httpRequest"}]}), encoding="utf-8")
    record = N8NIngester().parse_file(path)
    assert record["framework"] == "n8n"
    assert record["tools"] == ["webhook", "httpRequest"]
