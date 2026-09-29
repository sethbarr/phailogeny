"""Lineage classification helpers for homology vs convergence."""

from __future__ import annotations


def classify_lineage(lineage_similarity: float, functional_similarity: float, threshold_lineage: float = 0.7, threshold_functional: float = 0.7) -> str:
    """Classify a pair as homologous or convergent based on lineage and function signals."""
    if lineage_similarity >= threshold_lineage:
        return "homologous"
    if functional_similarity >= threshold_functional and lineage_similarity < threshold_lineage:
        return "convergent"
    return "unclassified"
