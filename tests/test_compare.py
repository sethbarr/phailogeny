"""Tests for comparing two analyses of the same agents, and for twin-based style offsets."""

from __future__ import annotations

import numpy as np
from skbio import DistanceMatrix
from skbio.tree import nj

from phailogeny.estate import fit_style_offsets, remove_style
from phailogeny.structure.compare import compare_analyses


def _analysis(distance: np.ndarray, labels: list[int]) -> dict:
    ids = [f"a{i}" for i in range(len(distance))]
    records = [{"source": "r0" if i < 3 else "r1"} for i in range(len(distance))]
    tree = nj(DistanceMatrix(distance, ids)).root_at_midpoint()
    return {"ids": ids, "records": records, "functional": distance, "tree": tree, "cluster_labels": labels}


def test_identical_analyses_compare_as_identical() -> None:
    points = np.random.default_rng(0).normal(size=(6, 2))
    distance = np.linalg.norm(points[:, None] - points[None], axis=2)
    a = _analysis(distance, [0, 0, 1, 1, -1, -1])
    result = compare_analyses(a, a)
    assert result["robinson_foulds_proportion"] == 0.0
    assert result["cophenetic_r"] == 1.0 and result["cluster_ari"] == 1.0
    assert result["nearest_neighbour_unchanged"] == 1.0


def test_style_offsets_recover_a_repo_shift_and_restore_twin_matches() -> None:
    rng = np.random.default_rng(1)
    roles = rng.normal(size=(20, 8))
    style = np.zeros(8)
    style[0] = 3.0
    vectors = np.vstack([roles, roles + style])  # repo b = same roles written in a different style
    sources = ["a"] * 20 + ["b"] * 20
    twins = [(i, i + 20) for i in range(20)]
    offsets = fit_style_offsets(vectors, sources, twins, ridge=0.1)
    assert np.allclose(offsets["b"] - offsets["a"], style, atol=0.1)
    cleaned = remove_style(vectors, sources, offsets)
    sims = cleaned[:20] @ cleaned[20:].T
    assert (sims.argmax(axis=1) == np.arange(20)).all()
