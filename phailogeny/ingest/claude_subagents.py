"""Parser for Claude Code subagent collections from GitHub repositories."""

from __future__ import annotations

from pathlib import Path

import yaml


class ClaudeSubagentIngester:
    """Read Claude subagent markdown definitions and normalize them into records."""

    def __init__(self, source: str = "claude-subagents") -> None:
        self.source = source

    def _split_frontmatter(self, text: str) -> tuple[dict[str, object], str]:
        """Split YAML frontmatter from the body content."""
        if not text.startswith("---\n"):
            return {}, text

        sections = text.split("\n---\n", 1)
        if len(sections) != 2:
            return {}, text

        frontmatter = sections[0][4:]
        body = sections[1]
        if not frontmatter.strip():
            return {}, body

        try:
            parsed = yaml.safe_load(frontmatter)
        except yaml.YAMLError:
            parsed = self._loose_frontmatter(frontmatter)
        if parsed is None:
            return {}, body
        if not isinstance(parsed, dict):
            return {}, body
        return parsed, body

    def _loose_frontmatter(self, frontmatter: str) -> dict[str, object]:
        """Read `key: value` lines when frontmatter is not valid YAML (e.g. unquoted colons)."""
        parsed: dict[str, object] = {}
        for line in frontmatter.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() and " " not in key.strip() and not line[:1].isspace():
                parsed[key.strip()] = value.strip()
        return parsed

    def parse_file(self, path: str | Path) -> dict[str, object]:
        """Parse a single subagent markdown file into a basic record."""
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(text)

        name = frontmatter.get("name") or file_path.stem
        description = frontmatter.get("description") or frontmatter.get("purpose") or ""
        tools = frontmatter.get("tools") or []
        model = frontmatter.get("model")
        metadata = {key: value for key, value in frontmatter.items() if key not in {"name", "description", "purpose", "tools", "model"}}

        if isinstance(tools, str):
            tools = [item.strip() for item in tools.split(",") if item.strip()]
        if not isinstance(tools, list):
            tools = []

        prompt = body.strip() if body.strip() else text.strip()

        return {
            "source": self.source,
            "source_path": str(file_path),
            "name": str(name),
            "purpose": str(description),
            "prompt": prompt,
            "tools": list(tools),
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": model,
            "framework": "claude-code",
            "metadata": metadata,
        }
