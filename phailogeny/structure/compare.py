"""Compare two analyses of the same agents (e.g. style-centred vs uncentred embeddings)."""

from __future__ import annotations

import numpy as np


def _nearest(distance: np.ndarray) -> np.ndarray:
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1)


def _mixed_clusters(labels: list[int], sources: np.ndarray) -> tuple[int, int]:
    clusters = [c for c in set(labels) if c >= 0]
    lab = np.array(labels)
    return len(clusters), sum(1 for c in clusters if len(set(sources[lab == c])) > 1)


def compare_analyses(a: dict, b: dict, name_a: str = "a", name_b: str = "b", examples: int = 15) -> dict:
    """Tree topology, tree distances, clusters and nearest neighbours of two analyses.

    - Robinson-Foulds (unrooted, proportion): share of splits found in one tree but not the other.
    - Cophenetic r: correlation of tip-to-tip path lengths between the trees.
    - Cluster ARI: agreement of HDBSCAN cluster assignments (1 = identical).
    """
    from sklearn.metrics import adjusted_rand_score

    if a["ids"] != b["ids"]:
        raise ValueError("analyses must cover the same agents in the same order")
    ids = a["ids"]
    sources = np.array([str(r["source"]) for r in a["records"]])
    nn_a, nn_b = _nearest(a["functional"]), _nearest(b["functional"])
    unitcorr = a["tree"].compare_cophenet(b["tree"], metric="unitcorr")
    changed = [
        {"agent": ids[i], f"nearest_{name_a}": ids[nn_a[i]], f"nearest_{name_b}": ids[nn_b[i]]}
        for i in np.where(nn_a != nn_b)[0]
    ]
    per_method = {}
    for name, result, nn in ((name_a, a, nn_a), (name_b, b, nn_b)):
        n_clusters, mixed = _mixed_clusters(result["cluster_labels"], sources)
        per_method[name] = {
            "same_repo_nearest_neighbour": round(float((sources[nn] == sources).mean()), 3),
            "clusters": n_clusters,
            "clusters_spanning_repos": mixed,
            "noise_agents": int(sum(1 for x in result["cluster_labels"] if x < 0)),
        }
    return {
        "methods": per_method,
        "robinson_foulds_proportion": round(float(a["tree"].compare_rfd(b["tree"], proportion=True, rooted=False)), 3),
        "cophenetic_r": round(float(1.0 - 2.0 * unitcorr), 3),
        "cluster_ari": round(float(adjusted_rand_score(a["cluster_labels"], b["cluster_labels"])), 3),
        "nearest_neighbour_unchanged": round(float((nn_a == nn_b).mean()), 3),
        "nearest_neighbour_changed_examples": changed[:examples],
    }
