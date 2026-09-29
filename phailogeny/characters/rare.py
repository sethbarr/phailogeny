"""Shared rare features: the synapomorphy-style signal for copying.

Two agents written independently for the same job share the job's common vocabulary. A copy,
even a full rewrite, tends to also keep idiosyncratic details of its source: an unusual library,
a specific threshold, an odd example. Those details are rare across the corpus, so sharing them is
evidence of ancestry rather than of function.

Rarity is judged with the pair itself excluded: a term counts as rare for a pair if at most
`max_others` other agents in the reference corpus use it. That keeps natural pairs (both members
in the corpus) and new-vs-corpus pairs on the same footing.
"""

from __future__ import annotations

import math
from collections import Counter

from phailogeny.characters.traits import normalise

STOPWORDS = set(
    """a an and are as at be been being but by can could did do does for from had has have how if in
    into is it its may might must no not of on or our shall should so such than that the their them
    then there these they this those to under up use used using was we were what when where which
    while who will with within without would you your yours via per each any all also more most
    other only own same very just both few many much new one two three etc e.g i.e""".split()
)


def terms(text: str) -> set[str]:
    """Content unigrams (>= 3 chars, not stopwords) plus bigrams of adjacent content words."""
    tokens = normalise(text).split()
    content = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
    unigrams = set(content)
    bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a in unigrams and b in unigrams}
    return unigrams | bigrams


class RareIndex:
    """Document frequencies over a reference corpus (a list of term sets)."""

    def __init__(self, corpus_terms: list[set[str]]) -> None:
        self.n = len(corpus_terms)
        self.df: Counter[str] = Counter(t for doc in corpus_terms for t in doc)

    def shared_rare(
        self,
        a: set[str],
        b: set[str],
        a_in_corpus: bool,
        b_in_corpus: bool,
        max_others: int = 3,
    ) -> tuple[float, list[str]]:
        """IDF-weighted count of terms both use and at most `max_others` other agents use.

        Returns (score, terms sorted rarest first). Score = sum of log(n / (1 + others)).
        """
        shared = []
        score = 0.0
        for term in a & b:
            others = self.df.get(term, 0) - int(a_in_corpus) - int(b_in_corpus)
            if others <= max_others:
                shared.append((others, term))
                score += math.log(self.n / (1 + others))
        return score, [t for _, t in sorted(shared)]
