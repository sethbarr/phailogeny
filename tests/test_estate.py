"""Tests for the estate pipeline pieces that don't need the embedding model."""

from __future__ import annotations

import math

import numpy as np

from phailogeny.estate import (
    assign_ids,
    combine_blocks,
    dedupe_copies,
    minhash_lineage_similarity,
    strip_corpus_boilerplate,
    weighted_set_distances,
)


def test_empty_tool_sets_are_missing_not_identical() -> None:
    result = weighted_set_distances([set(), set(), {"read"}])
    assert math.isnan(result[0, 1])
    assert result[2, 2] == 0.0


def test_combine_blocks_skips_missing_blocks_per_pair() -> None:
    purpose = np.array([[0.0, 0.2], [0.2, 0.0]])
    capability = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    combined = combine_blocks({"purpose": purpose, "capability": capability}, {"purpose": 1, "capability": 1})
    assert combined[0, 1] == 0.2


def test_lineage_separates_copied_from_rewritten_text() -> None:
    base = "you are an expert reviewer who checks code for bugs and security issues in every change"
    sims = minhash_lineage_similarity([base, base + " carefully", "audit the diff looking for defects and vulnerabilities"])
    assert sims[0, 1] > 0.7
    assert sims[0, 2] < 0.1


def test_dedupe_and_ids_are_unique() -> None:
    records = [
        {"source": "a", "name": "x", "purpose": "p", "prompt": "q", "source_path": "1"},
        {"source": "a", "name": "x", "purpose": "p", "prompt": "q", "source_path": "2"},
        {"source": "a", "name": "x", "purpose": "p2", "prompt": "q", "source_path": "3"},
    ]
    unique = dedupe_copies(records)
    assert len(unique) == 2 and unique[0]["copies"] == ["2"]
    assert assign_ids(unique) == ["a/x", "a/x~2"]


def test_boilerplate_shared_across_source_is_stripped() -> None:
    records = [{"source": "s", "prompt": f"shared line\nunique {i}"} for i in range(5)]
    assert strip_corpus_boilerplate(records, threshold=0.1)[0] == "unique 0"


def test_codex_toml_is_parsed(tmp_path) -> None:
    from phailogeny.estate import load_subagent_dirs

    repo = tmp_path / "codex"
    repo.mkdir()
    (repo / "reviewer.toml").write_text(
        'name = "reviewer"\ndescription = "Reviews code"\ndeveloper_instructions = """\nCheck diffs.\n"""\n',
        encoding="utf-8",
    )
    records = load_subagent_dirs([str(repo)])
    assert records[0]["prompt"] == "Check diffs." and records[0]["framework"] == "codex"
