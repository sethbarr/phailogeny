"""Corpus storage and deduplication helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a JSONL corpus from disk."""
    records: list[dict[str, object]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    """Write a corpus to JSONL."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def dedupe_exact_copies(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove exact-duplicate records while preserving order."""
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for record in records:
        payload = json.dumps(record, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        unique.append(record)
    return unique
