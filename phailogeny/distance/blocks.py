"""Distance functions for each semantic block in the phailogeny model."""

from __future__ import annotations

from collections import Counter


def _normalise_text(value: str) -> str:
    """Normalise text by lowercasing and collapsing whitespace."""
    return " ".join(str(value).lower().split())


def _token_set(value: str) -> set[str]:
    """Return a token set for text similarity checks."""
    return set(_normalise_text(value).split())


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    """Compute 1 - Jaccard similarity for two sets."""
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    similarity = len(left & right) / len(union)
    return 1.0 - similarity


def purpose_distance(left: str, right: str) -> float:
    """Distance between purpose strings using token overlap."""
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    return _jaccard_distance(left_tokens, right_tokens)


def prompt_distance(left: str, right: str) -> float:
    """Distance between prompt bodies using token overlap."""
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    return _jaccard_distance(left_tokens, right_tokens)


def capability_distance(
    left: set[str] | list[str] | tuple[str, ...],
    right: set[str] | list[str] | tuple[str, ...],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted Jaccard distance for tool or data-source capability sets."""
    left_set = {str(item) for item in left}
    right_set = {str(item) for item in right}
    if weights is None:
        weights = {item: 1.0 for item in left_set | right_set}

    all_items = set(weights) | left_set | right_set
    left_weights = {item: weights.get(item, 1.0) for item in all_items if item in left_set}
    right_weights = {item: weights.get(item, 1.0) for item in all_items if item in right_set}

    if not all_items:
        return 0.0

    min_weight = sum(min(left_weights.get(item, 0.0), right_weights.get(item, 0.0)) for item in all_items)
    max_weight = sum(max(left_weights.get(item, 0.0), right_weights.get(item, 0.0)) for item in all_items)
    if max_weight == 0:
        return 0.0
    similarity = min_weight / max_weight
    return 1.0 - similarity


def interface_distance(
    left: set[str] | list[str] | tuple[str, ...],
    right: set[str] | list[str] | tuple[str, ...],
) -> float:
    """Jaccard distance for interface modes such as input or output modes."""
    left_set = {str(item) for item in left}
    right_set = {str(item) for item in right}
    return _jaccard_distance(left_set, right_set)


def lineage_distance(left: str, right: str) -> float:
    """Distance based on raw text reuse across the prompt and purpose text."""
    left_tokens = Counter(_normalise_text(left).split())
    right_tokens = Counter(_normalise_text(right).split())

    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 1.0

    intersection = sum(min(left_tokens[token], right_tokens[token]) for token in set(left_tokens) | set(right_tokens))
    union = sum(max(left_tokens[token], right_tokens[token]) for token in set(left_tokens) | set(right_tokens))
    if union == 0:
        return 0.0
    similarity = intersection / union
    return 1.0 - similarity
