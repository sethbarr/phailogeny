"""A2A registry metadata ingest."""

from __future__ import annotations

import json
from pathlib import Path


class A2ARegistryIngester:
    """Load A2A registry card JSON into a normalized agent record."""

    def __init__(self, source: str = "a2a") -> None:
        self.source = source

    def parse_file(self, path: str | Path) -> dict[str, object]:
        """Parse a single A2A registry card JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        name = payload.get("name") or payload.get("agent_name") or ""
        description = payload.get("description") or payload.get("purpose") or ""
        skills = payload.get("skills") or []
        return {
            "source": self.source,
            "source_path": str(path),
            "name": str(name),
            "purpose": str(description),
            "prompt": str(payload.get("instructions") or payload.get("prompt") or ""),
            "tools": [str(skill.get("name") or skill) for skill in skills],
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": payload.get("model"),
            "framework": "a2a",
            "metadata": payload,
        }
