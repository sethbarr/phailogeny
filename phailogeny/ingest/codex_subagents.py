"""Parser for OpenAI Codex subagent definitions (TOML with developer_instructions)."""

from __future__ import annotations

import tomllib
from pathlib import Path


class CodexSubagentIngester:
    """Read Codex subagent .toml files and normalize them into records."""

    def __init__(self, source: str = "codex-subagents") -> None:
        self.source = source

    def parse_file(self, path: str | Path) -> dict[str, object]:
        """Parse a single Codex subagent TOML file into a basic record."""
        file_path = Path(path)
        payload = tomllib.loads(file_path.read_text(encoding="utf-8"))
        known = {"name", "description", "developer_instructions", "model"}
        return {
            "source": self.source,
            "source_path": str(file_path),
            "name": str(payload.get("name") or file_path.stem),
            "purpose": str(payload.get("description") or ""),
            "prompt": str(payload.get("developer_instructions") or "").strip(),
            "tools": [],
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": payload.get("model"),
            "framework": "codex",
            "metadata": {key: value for key, value in payload.items() if key not in known},
        }
