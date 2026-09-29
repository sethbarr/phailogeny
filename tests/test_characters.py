"""Tests for character extraction and module splitting."""

from __future__ import annotations

from phailogeny.characters.boilerplate import strip_boilerplate
from phailogeny.characters.modules import split_agent_modules


def test_strip_boilerplate_removes_common_lines() -> None:
    """Shared corpus boilerplate should be filtered from prompts."""
    lines = [
        "You are an expert assistant.",
        "You are an expert assistant.",
        "Answer carefully.",
    ]

    filtered = strip_boilerplate(lines, threshold=0.5)
    assert "You are an expert assistant." not in filtered
    assert "Answer carefully." in filtered


def test_split_agent_modules_handles_prompt_sections_and_tools() -> None:
    """Prompt sections and tool references should be extracted as modules."""
    agent = {
        "agent_id": "agent-1",
        "prompt": "# Context\nUse the files.\n\n# Rules\nBe careful.",
        "tools": ["read_file", "grep"],
    }

    modules = split_agent_modules(agent)
    kinds = {module.kind for module in modules}
    labels = {module.label for module in modules}

    assert "prompt_section" in kinds
    assert "tool" in kinds
    assert "section-0" in labels
    assert "read_file" in labels
