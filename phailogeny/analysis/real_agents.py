"""Real-agent corpus analysis for public agent definitions such as Claude subagents."""

from __future__ import annotations

from pathlib import Path

from phailogeny.distance.blocks import purpose_distance
from phailogeny.ingest.claude_subagents import ClaudeSubagentIngester


def load_real_agent_records(directory: str | Path, source: str = "claude-subagents") -> list[dict[str, object]]:
    """Load all markdown agent definitions in a directory and parse them into the corpus schema."""
    ingester = ClaudeSubagentIngester(source=source)
    records: list[dict[str, object]] = []
    directory_path = Path(directory)
    if not directory_path.exists():
        return records

    for path in sorted(directory_path.glob("**/*.md")):
        if path.name.startswith("."):
            continue
        record = ingester.parse_file(path)
        records.append(record)
    return records


def summarize_real_agents(records: list[dict[str, object]], top_n: int = 5) -> dict[str, object]:
    """Compute a lightweight similarity summary across a real-agent corpus."""
    if not records:
        return {"count": 0, "top_pairs": []}

    pairs: list[dict[str, object]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            left_name = str(left.get("name") or "unknown")
            right_name = str(right.get("name") or "unknown")
            distance = purpose_distance(str(left.get("purpose") or ""), str(right.get("purpose") or ""))
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "distance": distance,
                }
            )

    pairs.sort(key=lambda item: float(item["distance"]))
    return {
        "count": len(records),
        "top_pairs": pairs[:top_n],
        "mean_distance": sum(float(item["distance"]) for item in pairs) / len(pairs) if pairs else 0.0,
    }
