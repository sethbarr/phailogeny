"""Tests for structure-level clustering, lineage, and transfer helpers."""

from __future__ import annotations

from phailogeny.structure.cluster import cluster_agents
from phailogeny.structure.families import group_module_families
from phailogeny.structure.lineage import classify_lineage
from phailogeny.structure.transfer import detect_transfer_candidates


def test_cluster_agents_returns_labels_and_clusters() -> None:
    """The clustering helper should produce labels and cluster membership."""
    distances = [[0.0, 0.1], [0.1, 0.0]]
    result = cluster_agents(distances)
    assert result["labels"] == [0, 0]


def test_lineage_classifier_labels_homology_and_convergence() -> None:
    """The lineage classifier should separate high-lineage and high-function cases."""
    assert classify_lineage(0.9, 0.8) == "homologous"
    assert classify_lineage(0.2, 0.9) == "convergent"


def test_module_family_and_transfer_helpers_work() -> None:
    """Module grouping and transfer detection should produce stable outputs."""
    modules = [
        {"module_id": "m1", "label": "prompt-a", "content": "shared text"},
        {"module_id": "m2", "label": "prompt-b", "content": "shared text"},
    ]
    families = group_module_families(modules)
    candidates = detect_transfer_candidates(
        [("a", "b")],
        {"shared text": ["a", "b"]},
        vertical_distance={("a", "b"): 0.1},
        functional_distance={("a", "b"): 0.8},
    )

    assert "shared text" in families
    assert candidates[0]["status"] == "candidate-transfer"
