"""Paraphrase-lineage test: can each measure find an agent's true source after it is rewritten?

1. Sample wshobson agents (seeded) and have Claude rewrite each at three levels that mirror the
   copying seen in the wild: light edit, full rewrite, Codex-style checklist. Rewrites run through
   local Claude Code (sandboxed: no tools, no settings, empty temp dir), in parallel, and are
   stored append-only in runs/paraphrase/rewrites.jsonl keyed by model + level + prompt version +
   source hash, so reruns never regenerate.
2. Offline: add the rewrites to the corpus as new "repos" and rank the true source among all
   original agents under each measure. Agent names are stripped from all text first.

Run: .venv/bin/python scripts/paraphrase_test.py [--agents 50] [--workers 6] [--eval-only]
Writes out/paraphrase/results.json.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from phailogeny.analysis.measure_eval import same_repo_neighbour_rate
from phailogeny.characters.traits import _strip_name, load_taxonomy, trait_distances, trait_matrix
from phailogeny.estate import (
    dedupe_copies,
    embedding_vectors,
    fit_style_offsets,
    load_subagent_dirs,
    minhash_lineage_similarity,
    remove_style,
    strip_corpus_boilerplate,
    twin_pairs,
    vector_distances,
)
from phailogeny.phenotype.common_garden import claude_cli_call, sha

REPOS = ["wshobson", "voltagent", "lst97", "voltagent-codex", "buildwithclaude", "0xfurai",
         "contains-studio", "pentest", "vijaythecoder", "taches"]
MODEL = "claude-opus-5-5"
PROMPT_VERSION = 1
SYSTEM = "You rewrite AI agent definitions for a research study on how copied agents change."
LEVELS = {
    "light": (
        "Lightly edit this agent definition, as someone adapting it for their own collection would: "
        "keep its structure and most sentences, reword roughly a third of the sentences, and reorder "
        "a few list items. Keep the same role and responsibilities."
    ),
    "rewrite": (
        "Rewrite this agent definition from scratch for a different collection: same role, "
        "responsibilities and expertise, but entirely new wording, headings and structure. "
        "Do not reuse any phrase of five or more words."
    ),
    "codex": (
        "Convert this agent definition into a terse OpenAI Codex-style subagent: a one-sentence "
        "description, then at most 150 words of numbered 'Working mode' steps and a short list of "
        "what to return. Same role. Do not reuse any phrase of five or more words."
    ),
}
SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}},
    "required": ["description", "prompt"],
    "additionalProperties": False,
}
STORE = Path("runs/paraphrase/rewrites.jsonl")
OUT = Path("out/paraphrase")


def stem(record: dict) -> str:
    return Path(str(record["source_path"])).stem.lower()


def sample_sources(records: list[dict], n: int, seed: int = 11) -> list[int]:
    pool = [
        i for i, r in enumerate(records)
        if r["source"] == "wshobson" and len(str(r["prompt"]).split()) >= 150
    ]
    random.Random(seed).shuffle(pool)
    chosen, stems = [], set()
    for i in pool:
        if stem(records[i]) not in stems:
            chosen.append(i)
            stems.add(stem(records[i]))
        if len(chosen) == n:
            break
    return chosen


def rewrite_key(record: dict, level: str) -> str:
    return sha(f"{MODEL}|{level}|v{PROMPT_VERSION}|{record['purpose']}\n{record['prompt']}", 32)


def generate(records: list[dict], sources: list[int], workers: int) -> dict:
    done = {}
    if STORE.exists():
        done = {row["key"]: row for row in map(json.loads, STORE.read_text(encoding="utf-8").splitlines())}
    jobs = [(i, level) for i in sources for level in LEVELS if rewrite_key(records[i], level) not in done]
    lock = threading.Lock()
    status = {"queued": len(jobs), "done": 0, "failed": 0, "cost_usd": 0.0}
    STORE.parent.mkdir(parents=True, exist_ok=True)

    def work(i: int, level: str) -> tuple[int, str, dict]:
        record = records[i]
        task = (
            f"{LEVELS[level]}\n\nReturn the new description and the new system prompt.\n\n"
            f"<description>\n{record['purpose']}\n</description>\n\n<system_prompt>\n{record['prompt']}\n</system_prompt>"
        )
        return i, level, claude_cli_call(SYSTEM, task, MODEL, "low", SCHEMA)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(work, i, level) for i, level in jobs):
            i, level, result = future.result()
            with lock:
                if result.get("ok") and result.get("structured"):
                    row = {
                        "key": rewrite_key(records[i], level),
                        "source_agent": records[i]["agent_id"],
                        "level": level,
                        "prompt_version": PROMPT_VERSION,
                        "model": result.get("model"),
                        "cost_usd": result.get("cost_usd"),
                        "session_id": result.get("session_id"),
                        **result["structured"],
                    }
                    with STORE.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    status["done"] += 1
                    status["cost_usd"] += result.get("cost_usd") or 0.0
                else:
                    status["failed"] += 1
                    print("failed", records[i]["agent_id"], level, str(result.get("error"))[:200])
    status["cost_usd"] = round(status["cost_usd"], 3)
    return status


def evaluate(records: list[dict], sources: list[int]) -> dict:
    rows = [json.loads(line) for line in STORE.read_text(encoding="utf-8").splitlines()]
    by_agent = {r["agent_id"]: k for k, r in enumerate(records)}
    wanted = {rewrite_key(records[i], level) for i in sources for level in LEVELS}
    rows = [r for r in rows if r["key"] in wanted]
    n_orig = len(records)
    augmented = list(records) + [
        {
            "agent_id": f"paraphrase-{r['level']}/{records[by_agent[r['source_agent']]]['name']}",
            "name": records[by_agent[r["source_agent"]]]["name"],
            "source": f"paraphrase-{r['level']}",
            "source_path": records[by_agent[r["source_agent"]]]["source_path"],
            "purpose": r["description"],
            "prompt": r["prompt"],
        }
        for r in rows
    ]
    truth = [by_agent[r["source_agent"]] for r in rows]
    levels = [r["level"] for r in rows]
    src = [str(r["source"]) for r in augmented]

    def strip(r: dict, text: str) -> str:
        return _strip_name(_strip_name(text, str(r["name"])), stem(r))

    purposes = [strip(r, str(r["purpose"])) for r in augmented]
    prompts = [strip(r, p) for r, p in zip(augmented, strip_corpus_boilerplate(augmented))]
    P, Q = embedding_vectors(purposes, prompts)
    twins = [(i, j) for i, j in twin_pairs(records)]  # offsets learned from the original corpus only
    Pc = remove_style(P, src, fit_style_offsets(P[:n_orig], src[:n_orig], twins))
    Qc = remove_style(Q, src, fit_style_offsets(Q[:n_orig], src[:n_orig], twins))
    lineage_sim = minhash_lineage_similarity([f"{r['purpose']}\n{r['prompt']}" for r in augmented])
    taxonomy = load_taxonomy("data/traits.yaml")
    measures = {
        "text overlap (MinHash)": 1.0 - lineage_sim,
        "embeddings, uncentred": (vector_distances(P) + vector_distances(Q)) / 2,
        "embeddings, style-centred": (vector_distances(Pc) + vector_distances(Qc)) / 2,
        "keyword traits": trait_distances(trait_matrix(augmented, taxonomy)),
    }

    results: dict = {"n_sources": len(sources), "n_rewrites": len(rows), "levels": {}}
    for level in LEVELS:
        idx = [k for k, lv in enumerate(levels) if lv == level]
        rewrites = [n_orig + k for k in idx]
        entry = {
            "n": len(idx),
            "median_5gram_overlap_with_source": round(float(np.median([lineage_sim[a, truth[k]] for a, k in zip(rewrites, idx)])), 3),
            "flagged_as_copy (overlap >= 0.3)": round(float(np.mean([lineage_sim[a, truth[k]] >= 0.3 for a, k in zip(rewrites, idx)])), 3),
            "measures": {},
        }
        for name, distance in measures.items():
            ranks = []
            for a, k in zip(rewrites, idx):
                pool = distance[a, :n_orig]  # search among the original agents only
                better = int((pool < pool[truth[k]]).sum())
                tied = int((pool == pool[truth[k]]).sum()) - 1
                ranks.append(1 + better + tied / 2)
            ranks = np.array(ranks)
            entry["measures"][name] = {
                "source_found_first": round(float((ranks == 1).mean()), 3),
                "source_in_top5": round(float((ranks <= 5).mean()), 3),
                "median_rank_of_1003": float(np.median(ranks)),
            }
        results["levels"][level] = entry
    results["same_repo_nn_original_corpus"] = {
        name: round(same_repo_neighbour_rate(d[:n_orig, :n_orig], np.array(src[:n_orig])), 3) for name, d in measures.items()
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    records = dedupe_copies(load_subagent_dirs([f"data/raw/{d}" for d in REPOS]))
    from phailogeny.estate import assign_ids

    assign_ids(records)
    sources = sample_sources(records, args.agents)
    if not args.eval_only:
        print(json.dumps(generate(records, sources, args.workers)))
    results = evaluate(records, sources)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
