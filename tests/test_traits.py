"""Tests for local trait extraction and the twin-retrieval evaluation."""

from __future__ import annotations

import numpy as np

from phailogeny.analysis.measure_eval import twin_retrieval
from phailogeny.characters.traits import count_traits, normalise, trait_distances, trait_matrix

TAXONOMY = {
    "language": {"cpp": ["c++"], "csharp": ["c#", ".net"], "python": ["python"]},
    "activity": {"review": ["code review"], "debug": ["debug", "root cause"]},
}


def test_normalise_keeps_symbol_languages_and_drops_sentence_periods() -> None:
    assert " c++ " in normalise("Expert in C++.") and " .net " in normalise("Uses .NET, C#!")


def test_count_traits_matches_whole_tokens_only() -> None:
    counts = count_traits("Python code review; find the root cause. Pythonic!", TAXONOMY)
    assert counts == {"language:python": 1, "activity:review": 1, "activity:debug": 1}


def test_agent_name_is_not_used_for_traits() -> None:
    records = [{"name": "python-debug", "purpose": "python-debug helper", "prompt": ""}]
    assert trait_matrix(records, TAXONOMY).sum() == 0


def test_trait_distance_is_zero_for_identical_profiles() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    distance = trait_distances(matrix, idf=False)
    assert distance[0, 1] == 0.0 and distance[0, 2] == 1.0


def test_retrieval_ties_rank_mid_pool_not_first() -> None:
    distance = np.ones((4, 4))
    np.fill_diagonal(distance, 0.0)
    sources = np.array(["a", "b", "b", "b"])
    assert twin_retrieval(distance, sources, [(0, 1)])[0] == 2.0
