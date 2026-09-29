"""n8n workflow ingest for structured workflow JSON."""

from __future__ import annotations

import json
from pathlib import Path


class N8NIngester:
    """Load n8n workflow JSON into a normalized agent-like record."""

    def __init__(self, source: str = "n8n") -> None:
        self.source = source

    def parse_file(self, path: str | Path) -> dict[str, object]:
        """Parse an n8n workflow file into the project schema."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        name = payload.get("name") or "workflow"
        nodes = payload.get("nodes") or []
        tools = []
        for node in nodes:
            node_name = node.get("type") or node.get("name") or ""
            if node_name:
                tools.append(str(node_name))
        return {
            "source": self.source,
            "source_path": str(path),
            "name": str(name),
            "purpose": str(payload.get("description") or ""),
            "prompt": str(payload.get("workflow") or ""),
            "tools": tools,
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": None,
            "framework": "n8n",
            "metadata": payload,
        }
