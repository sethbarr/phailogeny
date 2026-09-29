"""Boilerplate stripping utilities for corpus text."""

from __future__ import annotations

from collections import Counter


def strip_boilerplate(lines: list[str], threshold: float = 0.10) -> list[str]:
    """Remove corpus-common lines above a document-frequency threshold.

    Args:
        lines: Text lines to normalize.
        threshold: Fraction of documents in which a line may appear before it's stripped.

    Returns:
        Filtered lines with boilerplate removed.
    """
    if not lines:
        return []

    counts = Counter()
    total = 0
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        counts[normalized] += 1
        total += 1

    if total == 0:
        return lines

    retained: list[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized:
            retained.append(line)
            continue
        if counts[normalized] / max(total, 1) <= threshold:
            retained.append(line)
    return retained
