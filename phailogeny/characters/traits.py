"""Declared-trait extraction: a fixed taxonomy matched against each agent's description and prompt.

Traits describe what an agent says it is for (languages, platforms, activities, behaviour rules),
independent of how the prompt is written. That makes them comparable across formats (a terse
Codex checklist and a long Claude essay for the same role should share traits). Runs locally with
string operations only; no model calls.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

KEEP = set("+#.")


def normalise(text: str) -> str:
    """Lowercase, turn punctuation (except + # .) into spaces, drop sentence-final periods."""
    chars = [c if c.isalnum() or c in KEEP else " " for c in text.lower()]
    tokens = [token.rstrip(".") for token in "".join(chars).split()]
    return " " + " ".join(token for token in tokens if token) + " "


def load_taxonomy(path: str | Path) -> dict[str, dict[str, list[str]]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["categories"]


def trait_names(taxonomy: dict) -> list[str]:
    return [f"{category}:{trait}" for category, traits in taxonomy.items() for trait in traits]


def count_traits(text: str, taxonomy: dict) -> Counter[str]:
    """Count phrase hits per trait on whole-token boundaries."""
    haystack = normalise(text)
    counts: Counter[str] = Counter()
    for category, traits in taxonomy.items():
        for trait, phrases in traits.items():
            hits = sum(haystack.count(normalise(str(phrase))) for phrase in phrases)
            if hits:
                counts[f"{category}:{trait}"] = hits
    return counts


def _strip_name(text: str, name: str) -> str:
    """Remove the agent's own name (and its hyphen/space variants) so it cannot drive matches."""
    out = text
    for variant in {name, name.replace("-", " "), name.replace("_", " ")}:
        if variant.strip():
            out = out.replace(variant, " ").replace(variant.lower(), " ")
    return out


def trait_matrix(records: list[dict], taxonomy: dict, description_weight: float = 3.0) -> np.ndarray:
    """Agents x traits. Score = log1p(weighted hits); description hits count 3x prompt hits."""
    names = trait_names(taxonomy)
    column = {name: k for k, name in enumerate(names)}
    matrix = np.zeros((len(records), len(names)))
    for i, record in enumerate(records):
        name = str(record.get("name") or "")
        description = count_traits(_strip_name(str(record.get("purpose") or ""), name), taxonomy)
        prompt = count_traits(_strip_name(str(record.get("prompt") or ""), name), taxonomy)
        for trait in set(description) | set(prompt):
            matrix[i, column[trait]] = math.log1p(description_weight * description[trait] + prompt[trait])
    return matrix


def trait_distances(matrix: np.ndarray, idf: bool = True, categories: list[str] | None = None, names: list[str] | None = None) -> np.ndarray:
    """IDF-weighted Jaccard distance on trait scores (common traits count less)."""
    data = matrix
    if categories is not None and names is not None:
        keep = [k for k, n in enumerate(names) if n.split(":")[0] in categories]
        data = matrix[:, keep]
    n = data.shape[0]
    if idf:
        doc_freq = (data > 0).sum(axis=0)
        data = data * (np.log((n + 1) / (doc_freq + 1)) + 1.0)
    distance = np.ones((n, n))
    for i in range(n):
        mins = np.minimum(data[i], data).sum(axis=1)
        maxs = np.maximum(data[i], data).sum(axis=1)
        distance[i] = np.where(maxs > 0, 1.0 - mins / np.where(maxs > 0, maxs, 1), 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance
