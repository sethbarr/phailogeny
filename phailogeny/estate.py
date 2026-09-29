"""End-to-end estate analysis: records -> block distances -> NJ tree, clusters and pairs."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

from phailogeny.ingest.claude_subagents import ClaudeSubagentIngester
from phailogeny.ingest.codex_subagents import CodexSubagentIngester
from phailogeny.structure.lineage import classify_lineage

SKIP_NAMES = {"readme.md", "claude.md", "skill.md", "contributing.md", "license.md", "changelog.md"}
DEFAULT_WEIGHTS = {"purpose": 1.0, "prompt": 1.0, "capability": 1.0, "interface": 1.0}


def load_subagent_dirs(directories: list[str]) -> list[dict[str, object]]:
    """Load Claude subagent markdown files from one or more repo directories.

    Each directory's name becomes the source label. When a directory holds any files under an
    `agents/` folder, only those are kept (repos like wshobson also ship skills and commands).
    Codex `.toml` files carrying `developer_instructions` are read with the Codex parser.
    Files without a description are skipped.
    """
    records: list[dict[str, object]] = []
    claude, codex = ClaudeSubagentIngester(), CodexSubagentIngester()
    for directory in directories:
        root = Path(directory)
        label = root.name
        paths = [p for p in sorted(root.glob("**/*.md")) if p.name.lower() not in SKIP_NAMES]
        paths += [p for p in sorted(root.glob("**/*.toml")) if "developer_instructions" in p.read_text(encoding="utf-8")]
        in_agents = [p for p in paths if "agents" in p.relative_to(root).parts[:-1]]
        if in_agents:
            paths = in_agents
        for path in paths:
            ingester = codex if path.suffix == ".toml" else claude
            try:
                record = ingester.parse_file(path)
            except Exception:
                continue
            if not record["purpose"]:
                continue
            record["source"] = label
            records.append(record)
    return records


def dedupe_copies(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse agents whose purpose and prompt are identical, remembering where the copies live."""
    kept: dict[str, dict[str, object]] = {}
    for record in records:
        key = hashlib.sha1(f"{record['purpose']}\n{record['prompt']}".encode()).hexdigest()
        if key in kept:
            kept[key].setdefault("copies", []).append(str(record["source_path"]))
            continue
        kept[key] = record
    return list(kept.values())


def assign_ids(records: list[dict[str, object]]) -> list[str]:
    """Give each record a unique, Newick-safe leaf label of the form source/name."""
    ids: list[str] = []
    seen: Counter[str] = Counter()
    for record in records:
        base = f"{record['source']}/{record['name']}".replace(" ", "_")
        for char in "():;,'[]":
            base = base.replace(char, "_")
        seen[base] += 1
        label = base if seen[base] == 1 else f"{base}~{seen[base]}"
        record["agent_id"] = label
        ids.append(label)
    return ids


def strip_corpus_boilerplate(records: list[dict[str, object]], threshold: float = 0.10) -> list[str]:
    """Drop prompt lines that appear in more than `threshold` of agents from the same source."""
    by_source: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_source.setdefault(str(record["source"]), []).append(index)

    stripped = [""] * len(records)
    for indices in by_source.values():
        doc_freq: Counter[str] = Counter()
        for index in indices:
            lines = {line.strip().lower() for line in str(records[index]["prompt"]).splitlines()}
            doc_freq.update(line for line in lines if line)
        limit = max(threshold * len(indices), 2)
        for index in indices:
            keep = [
                line
                for line in str(records[index]["prompt"]).splitlines()
                if line.strip() and doc_freq[line.strip().lower()] <= limit
            ]
            stripped[index] = "\n".join(keep)
    return stripped


def _chunks(text: str, words_per_chunk: int = 150) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def _cosine_distance(vectors: np.ndarray, present: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1, norms)
    distance = np.clip(1.0 - unit @ unit.T, 0.0, 1.0)
    distance[~present, :] = np.nan
    distance[:, ~present] = np.nan
    return distance


def embedding_vectors(
    purposes: list[str], prompts: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors for each purpose, and for each prompt (mean of 150-word chunk embeddings).

    Rows for empty text are all zeros.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    purpose_vecs = np.asarray(model.encode(purposes, normalize_embeddings=True, show_progress_bar=False))
    purpose_vecs[[not p.strip() for p in purposes]] = 0.0

    all_chunks: list[str] = []
    owner: list[int] = []
    for index, prompt in enumerate(prompts):
        for chunk in _chunks(prompt):
            all_chunks.append(chunk)
            owner.append(index)
    prompt_vecs = np.zeros((len(prompts), purpose_vecs.shape[1]))
    if all_chunks:
        chunk_vecs = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
        np.add.at(prompt_vecs, np.array(owner), chunk_vecs)
    return purpose_vecs, _unit(prompt_vecs)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def vector_distances(vectors: np.ndarray) -> np.ndarray:
    """1 - cosine between unit vectors; NaN for agents with no text."""
    return _cosine_distance(vectors, np.linalg.norm(vectors, axis=1) > 0)


def embedding_distances(
    purposes: list[str], prompts: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> tuple[np.ndarray, np.ndarray]:
    """Purpose and prompt distance matrices (1 - cosine) from a local sentence-transformers model."""
    purpose_vecs, prompt_vecs = embedding_vectors(purposes, prompts, model_name)
    return vector_distances(purpose_vecs), vector_distances(prompt_vecs)


def twin_pairs(records: list[dict[str, object]]) -> list[tuple[int, int]]:
    """Cross-repo pairs of agents that share a filename (e.g. two `debugger.md`)."""
    by_stem: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_stem.setdefault(Path(str(record["source_path"])).stem.lower(), []).append(index)
    return [
        (i, j)
        for members in by_stem.values()
        for a, i in enumerate(members)
        for j in members[a + 1 :]
        if records[i]["source"] != records[j]["source"]
    ]


def fit_style_offsets(
    vectors: np.ndarray, sources: list[str], pairs: list[tuple[int, int]], ridge: float = 5.0
) -> dict[str, np.ndarray]:
    """Per-repo style offsets learned from twins.

    Twins share a job, so the gap between them is mostly writing style. Solve
    min sum ||(x_i - x_j) - (o_s - o_t)||^2 + ridge * sum ||o_s||^2 over twin pairs (i in repo s,
    j in repo t). A repo with few twins gets an offset shrunk toward zero, so a single-domain repo
    keeps its domain.
    """
    repos = sorted(set(sources))
    index = {repo: k for k, repo in enumerate(repos)}
    lhs = ridge * np.eye(len(repos))
    rhs = np.zeros((len(repos), vectors.shape[1]))
    for i, j in pairs:
        s, t = index[sources[i]], index[sources[j]]
        diff = vectors[i] - vectors[j]
        lhs[s, s] += 1
        lhs[t, t] += 1
        lhs[s, t] -= 1
        lhs[t, s] -= 1
        rhs[s] += diff
        rhs[t] -= diff
    offsets = np.linalg.solve(lhs, rhs)
    return {repo: offsets[k] for repo, k in index.items()}


def remove_style(vectors: np.ndarray, sources: list[str], offsets: dict[str, np.ndarray]) -> np.ndarray:
    present = np.linalg.norm(vectors, axis=1) > 0
    shifted = vectors - np.array([offsets.get(src, 0.0) for src in sources])
    shifted[~present] = 0.0
    return _unit(shifted)


def weighted_set_distances(sets: list[set[str]], idf: bool = True) -> np.ndarray:
    """Weighted Jaccard distance on sets; IDF weights make near-universal items count less.

    Pairs where either side is empty are NaN (missing), since an empty tool list usually means
    "not declared" rather than "no tools".
    """
    n = len(sets)
    doc_freq: Counter[str] = Counter(item for items in sets for item in items)
    weight = {item: (math.log((n + 1) / (count + 1)) + 1.0 if idf else 1.0) for item, count in doc_freq.items()}
    result = np.full((n, n), np.nan)
    for i in range(n):
        if not sets[i]:
            continue
        for j in range(i, n):
            if not sets[j]:
                continue
            inter = sum(weight[item] for item in sets[i] & sets[j])
            union = sum(weight[item] for item in sets[i] | sets[j])
            result[i, j] = result[j, i] = 1.0 - inter / union if union else 0.0
    return result


def minhash_lineage_similarity(texts: list[str], shingle: int = 5, num_perm: int = 128) -> np.ndarray:
    """Estimated Jaccard similarity of word 5-shingles: the literal text-reuse (lineage) signal."""
    from datasketch import MinHash

    signatures = []
    for text in texts:
        words = " ".join(text.lower().split()).split()
        minhash = MinHash(num_perm=num_perm, seed=1)
        for i in range(max(len(words) - shingle + 1, 1)):
            minhash.update(" ".join(words[i : i + shingle]).encode())
        signatures.append(minhash.hashvalues)
    sig = np.asarray(signatures)
    return (sig[:, None, :] == sig[None, :, :]).mean(axis=2)


def combine_blocks(blocks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Gower-style combination: per pair, average only the blocks that have data."""
    total = None
    weight_sum = None
    for name, matrix in blocks.items():
        w = weights.get(name, 1.0)
        present = ~np.isnan(matrix)
        contrib = np.where(present, matrix * w, 0.0)
        total = contrib if total is None else total + contrib
        weight_sum = present * w if weight_sum is None else weight_sum + present * w
    combined = np.where(weight_sum > 0, total / np.where(weight_sum > 0, weight_sum, 1), 1.0)
    combined = (combined + combined.T) / 2.0
    np.fill_diagonal(combined, 0.0)
    return combined


def analyse_estate(
    records: list[dict[str, object]],
    weights: dict[str, float] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    style_centring: bool = True,
    ridge: float = 5.0,
) -> dict[str, object]:
    """Run the genotype pipeline on normalised records and return matrices, tree and clusters.

    With `style_centring`, per-repo writing-style offsets learned from cross-repo twins are removed
    from the embeddings before distances are computed (see `fit_style_offsets`).
    """
    from sklearn.cluster import HDBSCAN
    from skbio import DistanceMatrix
    from skbio.tree import nj

    weights = weights or DEFAULT_WEIGHTS
    ids = assign_ids(records)
    purposes = [str(r.get("purpose") or "") for r in records]
    prompts = strip_corpus_boilerplate(records)

    purpose_v, prompt_v = embedding_vectors(purposes, prompts, model_name)
    sources = [str(r["source"]) for r in records]
    twins = twin_pairs(records)
    if style_centring and twins:
        purpose_v = remove_style(purpose_v, sources, fit_style_offsets(purpose_v, sources, twins, ridge))
        prompt_v = remove_style(prompt_v, sources, fit_style_offsets(prompt_v, sources, twins, ridge))
    purpose_d, prompt_d = vector_distances(purpose_v), vector_distances(prompt_v)
    tool_sets = [{str(t).strip().lower() for t in r.get("tools") or [] if str(t).strip()} for r in records]
    mode_sets = [set(r.get("input_modes") or []) | set(r.get("output_modes") or []) for r in records]
    blocks = {
        "purpose": purpose_d,
        "prompt": prompt_d,
        "capability": weighted_set_distances(tool_sets),
        "interface": weighted_set_distances(mode_sets, idf=False),
    }
    coverage = {name: float(np.mean(~np.isnan(m))) for name, m in blocks.items()}
    functional = combine_blocks(blocks, weights)
    lineage = minhash_lineage_similarity([f"{p}\n{r['prompt']}" for p, r in zip(purposes, records)])

    tree = nj(DistanceMatrix(functional, ids)).root_at_midpoint()

    labels = HDBSCAN(metric="precomputed", min_cluster_size=2, copy=True).fit_predict(functional)

    return {
        "ids": ids,
        "records": records,
        "blocks": blocks,
        "coverage": coverage,
        "weights": weights,
        "functional": functional,
        "style_centring": {"enabled": bool(style_centring and twins), "ridge": ridge, "twin_pairs": len(twins)},
        "lineage": lineage,
        "tree": tree,
        "cluster_labels": [int(x) for x in labels],
    }


def nearest_pairs(result: dict[str, object], limit: int = 50) -> list[dict[str, object]]:
    """Most functionally similar pairs, each tagged homologous (copied) or convergent."""
    ids = result["ids"]
    functional = result["functional"]
    lineage = result["lineage"]
    n = len(ids)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(functional[iu])[:limit]
    pairs = []
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        f_sim = 1.0 - float(functional[i, j])
        l_sim = float(lineage[i, j])
        pairs.append(
            {
                "left": ids[i],
                "right": ids[j],
                "functional_similarity": round(f_sim, 3),
                "lineage_similarity": round(l_sim, 3),
                "relation": classify_lineage(l_sim, f_sim, threshold_lineage=0.3, threshold_functional=0.6),
            }
        )
    return pairs


def nearest_neighbour_of_each(result: dict[str, object]) -> dict[str, dict[str, object]]:
    """For every agent, its single closest other agent."""
    ids = result["ids"]
    functional = result["functional"].copy()
    np.fill_diagonal(functional, np.inf)
    out = {}
    for i, agent in enumerate(ids):
        j = int(np.argmin(functional[i]))
        out[agent] = {"neighbour": ids[j], "functional_similarity": round(1.0 - float(functional[i, j]), 3)}
    return out
