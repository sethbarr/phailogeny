"""Compare similarity measures on known relationships, without any model calls.

Ground truth proxy: cross-repo agents that share a filename ("twins", e.g. two debugger agents).
Same name is imperfect (it doesn't guarantee the same job), so this ranks measures relative to
each other; the common-garden behaviour run is the independent check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from phailogeny.estate import twin_pairs

RELATIONS = {
    ("voltagent", "voltagent-codex"): "format change (same org, Claude -> Codex)",
    ("lst97", "wshobson"): "full rewrite (documented ancestry)",
    ("buildwithclaude", "wshobson"): "edited copy",
}


def relation_of(records: list[dict], i: int, j: int) -> str:
    key = tuple(sorted((str(records[i]["source"]), str(records[j]["source"]))))
    return RELATIONS.get(key, "other repos")


def twin_auc(distance: np.ndarray, sources: np.ndarray, twins: list[tuple[int, int]]) -> float:
    """P(a twin pair is closer than a random cross-repo non-twin pair). 0.5 = chance."""
    iu = np.triu_indices(len(sources), k=1)
    cross = sources[iu[0]] != sources[iu[1]]
    values = distance[iu][cross]
    twin_mask = np.zeros(len(sources) * len(sources), dtype=bool)
    for i, j in twins:
        twin_mask[min(i, j) * len(sources) + max(i, j)] = True
    is_twin = twin_mask[(iu[0] * len(sources) + iu[1])[cross]]
    ranks = rankdata(values)
    n_t, n_o = is_twin.sum(), (~is_twin).sum()
    # Mann-Whitney U on "smaller distance is better".
    u = ranks[is_twin].sum() - n_t * (n_t + 1) / 2
    return float(1.0 - u / (n_t * n_o))


def twin_retrieval(distance: np.ndarray, sources: np.ndarray, twins: list[tuple[int, int]]) -> list[float]:
    """For each twin, its rank among all agents of the twin's repo (1 = found first). Both directions.

    Ties take the average position, so a measure that scores everything 1.0 ranks mid-pool,
    not first.
    """
    ranks = []
    for i, j in twins:
        for a, b in ((i, j), (j, i)):
            pool = distance[a, np.where(sources == sources[b])[0]]
            better = int((pool < distance[a, b]).sum())
            tied_others = int((pool == distance[a, b]).sum()) - 1
            ranks.append(1 + better + tied_others / 2)
    return ranks


def same_repo_neighbour_rate(distance: np.ndarray, sources: np.ndarray) -> float:
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    return float((sources[d.argmin(axis=1)] == sources).mean())


def evaluate(records: list[dict], measures: dict[str, np.ndarray]) -> dict:
    sources = np.array([str(r["source"]) for r in records])
    twins = twin_pairs(records)
    groups: dict[str, list[tuple[int, int]]] = {}
    for i, j in twins:
        groups.setdefault(relation_of(records, i, j), []).append((i, j))

    out: dict[str, dict] = {}
    for name, distance in measures.items():
        row = {
            "auc_all_twins": round(twin_auc(distance, sources, twins), 3),
            "top1_all": round(float(np.mean(np.array(twin_retrieval(distance, sources, twins)) == 1)), 3),
            "same_repo_nn": round(same_repo_neighbour_rate(distance, sources), 3),
        }
        for relation, pairs in groups.items():
            ranks = np.array(twin_retrieval(distance, sources, pairs))
            row[f"top1 | {relation}"] = round(float(np.mean(ranks == 1)), 3)
            row[f"median rank | {relation}"] = float(np.median(ranks))
        out[name] = row
    return {"n_twin_pairs": len(twins), "groups": {k: len(v) for k, v in groups.items()}, "measures": out}
