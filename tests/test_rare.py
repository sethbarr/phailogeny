"""Tests for shared-rare-feature scoring."""

from __future__ import annotations

from phailogeny.characters.rare import RareIndex, terms


def test_terms_drop_stopwords_and_keep_content_bigrams() -> None:
    found = terms("Use Dapper for the data access layer")
    assert "dapper" in found and "access layer" in found and "the" not in found


def test_common_terms_are_not_rare_and_pair_is_excluded_from_its_own_rarity() -> None:
    corpus = [{"api", "dapper"}, {"api", "dapper"}, {"api"}, {"api"}, {"api"}]
    index = RareIndex(corpus)
    score, shared = index.shared_rare(corpus[0], corpus[1], True, True, max_others=0)
    assert shared == ["dapper"] and score > 0  # "api" is used by 3 other agents, so not rare
    _, shared_new = index.shared_rare({"dapper"}, corpus[0], False, True, max_others=0)
    assert shared_new == []  # one other corpus agent (corpus[1]) also uses it
