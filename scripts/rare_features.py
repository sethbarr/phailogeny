"""Do copies share rare features that independent same-job agents don't? (offline, no model calls)

Compares shared-rare-term scores for:
  - paraphrase rewrites vs their true source (known copies; from scripts/paraphrase_test.py)
  - natural cross-repo twins, split by documented relationship
  - close non-twin pairs (same job, different name) and random pairs (controls)
Then asks, at matched functional similarity, whether rare sharing separates known copies from
presumed-independent twins better than functional similarity alone.

Run after scripts/paraphrase_test.py: .venv/bin/python scripts/rare_features.py
Writes out/rare_features/results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import paraphrase_test as pt  # noqa: E402

from phailogeny.analysis.measure_eval import relation_of  # noqa: E402
from phailogeny.characters.rare import RareIndex, terms  # noqa: E402
from phailogeny.characters.traits import _strip_name  # noqa: E402
from phailogeny.estate import (  # noqa: E402
    assign_ids, dedupe_copies, embedding_vectors, fit_style_offsets, load_subagent_dirs,
    remove_style, strip_corpus_boilerplate, twin_pairs, vector_distances,
)

OUT = Path("out/rare_features")


def main() -> None:
    records = dedupe_copies(load_subagent_dirs([f"data/raw/{d}" for d in pt.REPOS]))
    assign_ids(records)
    n = len(records)
    by_id = {r["agent_id"]: k for k, r in enumerate(records)}
    rows = [json.loads(line) for line in pt.STORE.read_text(encoding="utf-8").splitlines()]
    rewrites = [
        {
            "name": records[by_id[r["source_agent"]]]["name"],
            "source": f"paraphrase-{r['level']}",
            "source_path": records[by_id[r["source_agent"]]]["source_path"],
            "purpose": r["description"],
            "prompt": r["prompt"],
            "level": r["level"],
            "truth": by_id[r["source_agent"]],
        }
        for r in rows
    ]
    augmented = records + rewrites

    def strip(r: dict, text: str) -> str:
        return _strip_name(_strip_name(text, str(r["name"])), pt.stem(r))

    # Rare terms from description + prompt, names removed; DF from the original corpus only.
    term_sets = [terms(strip(r, f"{r['purpose']}\n{r['prompt']}")) for r in augmented]
    index = RareIndex(term_sets[:n])

    # Functional similarity: style-centred embeddings (offsets learned from original twins).
    sources = [str(r["source"]) for r in augmented]
    purposes = [strip(r, str(r["purpose"])) for r in augmented]
    prompts = [strip(r, p) for r, p in zip(augmented, strip_corpus_boilerplate(augmented))]
    P, Q = embedding_vectors(purposes, prompts)
    twins = twin_pairs(records)
    P = remove_style(P, sources, fit_style_offsets(P[:n], sources[:n], twins))
    Q = remove_style(Q, sources, fit_style_offsets(Q[:n], sources[:n], twins))
    similarity = 1.0 - (vector_distances(P) + vector_distances(Q)) / 2

    groups: dict[str, list[tuple[int, int]]] = {}
    for k, r in enumerate(rewrites):
        groups.setdefault(f"known copy: paraphrase {r['level']}", []).append((n + k, r["truth"]))
    for i, j in twins:
        groups.setdefault(f"natural twins: {relation_of(records, i, j)}", []).append((i, j))
    twin_set = {frozenset(p) for p in twins}
    rng = np.random.default_rng(0)
    iu = np.triu_indices(n, k=1)
    cross = np.array(sources[:n])[iu[0]] != np.array(sources[:n])[iu[1]]
    sims = similarity[:n, :n][iu]
    close = np.where(cross & (sims >= 0.6))[0]
    groups["control: close non-twin pairs (sim >= 0.6)"] = [
        (int(iu[0][k]), int(iu[1][k])) for k in rng.choice(close, min(500, len(close)), replace=False)
        if frozenset((int(iu[0][k]), int(iu[1][k]))) not in twin_set
    ]
    rand = np.where(cross)[0]
    groups["control: random cross-repo pairs"] = [
        (int(iu[0][k]), int(iu[1][k])) for k in rng.choice(rand, 2000, replace=False)
    ]

    def score(i: int, j: int) -> tuple[float, list[str]]:
        return index.shared_rare(term_sets[i], term_sets[j], i < n, j < n)

    summary, pair_scores = {}, {}
    for name, pairs in groups.items():
        values = [score(i, j) for i, j in pairs]
        s = np.array([v[0] for v in values])
        f = np.array([similarity[i, j] for i, j in pairs])
        pair_scores[name] = (s, f)
        examples = [
            {"a": augmented[i].get("agent_id", f"{augmented[i]['source']}/{augmented[i]['name']}"),
             "b": augmented[j].get("agent_id"), "shared_rare": v[1][:8]}
            for (i, j), v in list(zip(pairs, values))[:3]
        ]
        summary[name] = {
            "n": len(pairs),
            "functional_similarity_median": round(float(np.median(f)), 3),
            "rare_score_median": round(float(np.median(s)), 2),
            "rare_score_p90": round(float(np.percentile(s, 90)), 2),
            "share_with_any_rare_term": round(float((s > 0).mean()), 3),
            "examples": examples,
        }

    # Can rare sharing separate known copies from presumed-independent twins at matched function?
    independent_s, independent_f = pair_scores["natural twins: other repos"]
    separation = {}
    for level in ("rewrite", "codex"):
        copy_s, copy_f = pair_scores[f"known copy: paraphrase {level}"]
        # Match each copy to independent twins within +/-0.03 functional similarity.
        matched = [k for f in copy_f for k in np.where(np.abs(independent_f - f) <= 0.03)[0]]
        labels = np.r_[np.ones(len(copy_s)), np.zeros(len(matched))]
        separation[level] = {
            "matched_independent_pairs": len(matched),
            "auc_rare_sharing": round(float(roc_auc_score(labels, np.r_[copy_s, independent_s[matched]])), 3),
            "auc_functional_similarity": round(float(roc_auc_score(labels, np.r_[copy_f, independent_f[matched]])), 3),
        }
    results = {"max_others": 3, "groups": summary, "copy_vs_independent_at_matched_similarity": separation}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, v in summary.items():
        print(f"{name:<52} n={v['n']:4d} func {v['functional_similarity_median']:.2f}  rare median {v['rare_score_median']:6.1f}  p90 {v['rare_score_p90']:6.1f}  any {v['share_with_any_rare_term']:.2f}")
    print(json.dumps(separation, indent=1))
    for name in ("known copy: paraphrase rewrite", "known copy: paraphrase codex", "natural twins: other repos"):
        print(name, summary[name]["examples"][:2])


if __name__ == "__main__":
    main()
