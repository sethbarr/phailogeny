"""Metadata ingest for the GPTZoo corpus."""

from __future__ import annotations

import json
from pathlib import Path


class GPTZooIngester:
    """Load GPTZoo-style JSON metadata into the phailogeny schema."""

    def __init__(self, source: str = "gptzoo") -> None:
        self.source = source

    def parse_file(self, path: str | Path) -> dict[str, object]:
        """Parse an individual GPTZoo metadata JSON object into a normalized record."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if not payload:
                return {
                    "source": self.source,
                    "source_path": str(path),
                    "name": "",
                    "purpose": "",
                    "prompt": "",
                    "tools": [],
                    "data_sources": [],
                    "permissions": [],
                    "input_modes": [],
                    "output_modes": [],
                    "model": None,
                    "framework": "gpt",
                    "metadata": {},
                }
            payload = payload[0]

        name = payload.get("name") or payload.get("title") or ""
        description = payload.get("description") or payload.get("purpose") or ""
        tools = payload.get("tools") or payload.get("tool_flags") or []
        return {
            "source": self.source,
            "source_path": str(path),
            "name": str(name),
            "purpose": str(description),
            "prompt": str(payload.get("instructions") or payload.get("prompt") or ""),
            "tools": [str(item) for item in tools],
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": payload.get("model"),
            "framework": "gpt",
            "metadata": payload,
        }
