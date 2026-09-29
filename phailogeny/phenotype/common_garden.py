"""Common-garden phenotype experiment: run every sampled agent on the same probe battery.

Traceability model
- `runs/store/responses.jsonl` and `runs/store/judgments.jsonl` are append-only and shared by all
  runs. Each row is keyed by a hash of everything that determines it (model, settings, the agent's
  system prompt, the probe text, the replicate), so a paid response is never requested twice and
  an edited probe or prompt can never be mixed with an old answer.
- `runs/<run_id>/` freezes one experiment: manifest (settings, code hash, probe hashes, sampling),
  agents.jsonl (full system prompts), genotype.json (distances at plan time), requests.jsonl,
  batches.jsonl (every batch submitted), analysis.json and report.md.
- `runs/index.jsonl` gets one summary line per analysed run, for comparing runs.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

RUNS_DIR = Path("runs")
JUDGE_VERSION = 1
BASELINE_ID = "__baseline__"
# Claude Code's -p mode needs some system prompt; the baseline gets this neutral one.
BASELINE_SYSTEM = "You are a helpful assistant."
# List prices, $ per million tokens (input, output). Thinking tokens bill as output.
LIST_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Batch prices are half of list.
BATCH_PRICES = {
    "claude-opus-5-5": (2.0, 10.0),
    "claude-sonnet-5-5": (1.0, 5.0),
    "claude-haiku-4-5": (0.5, 2.5),
}
JUDGE_FIELDS = [
    "asked_clarifying_question",
    "declined_or_redirected",
    "pushed_back",
    "wrote_code",
    "wrote_tests",
    "flagged_security",
]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{field: {"type": "boolean"} for field in JUDGE_FIELDS},
        "caught_planted_issue": {"type": "string", "enum": ["yes", "partly", "no", "n/a"]},
        "approach": {"type": "string"},
    },
    "required": [*JUDGE_FIELDS, "caught_planted_issue", "approach"],
    "additionalProperties": False,
}


def sha(text: str, length: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_probes(path: str | Path) -> dict:
    """Load the probe battery and stamp each probe with a hash of its text and planted issue."""
    battery = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for probe in battery["probes"]:
        probe["hash"] = sha(f"{probe['prompt']}\n{probe.get('planted')}", 16)
    return battery


def response_key(model: str, settings: dict, system: str | None, probe: dict, rep: int) -> str:
    """Stable id for one agent-probe-replicate call; doubles as the batch custom_id."""
    payload = json.dumps(
        {"model": model, "settings": settings, "system": sha(system or ""), "probe": probe["hash"], "rep": rep},
        sort_keys=True,
    )
    return sha(payload, 32)


def judge_key(judge_model: str, response_key_: str) -> str:
    return sha(f"{judge_model}|v{JUDGE_VERSION}|{response_key_}", 32)


# ---------------------------------------------------------------- sampling


def _stem(record: dict) -> str:
    return Path(str(record["source_path"])).stem.lower()


def select_agents(records: list[dict], functional: np.ndarray, n_agents: int, seed: int) -> dict:
    """Stratified sample: cross-repo same-name twins, close non-twin pairs, then random agents."""
    rng = random.Random(seed)
    n = len(records)
    stems = [_stem(r) for r in records]
    sources = [str(r["source"]) for r in records]

    twins = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if stems[i] == stems[j] and sources[i] != sources[j]
    ]
    rng.shuffle(twins)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(functional[iu])
    near = [
        (int(iu[0][k]), int(iu[1][k]))
        for k in order[: 20 * n_agents]
        if stems[iu[0][k]] != stems[iu[1][k]]
    ]
    rng.shuffle(near)

    chosen: list[int] = []
    strata: dict[int, str] = {}
    pairs: dict[str, list[list[int]]] = {"twin": [], "near": []}

    def take(pair_list: list[tuple[int, int]], label: str, budget: int) -> None:
        for i, j in pair_list:
            if len(chosen) + 2 > budget:
                return
            if i in strata or j in strata:
                continue
            for index in (i, j):
                chosen.append(index)
                strata[index] = label
            pairs[label].append([i, j])

    take(twins, "twin", int(n_agents * 0.4))
    take(near, "near", int(n_agents * 0.7))
    rest = [i for i in range(n) if i not in strata]
    rng.shuffle(rest)
    for index in rest[: n_agents - len(chosen)]:
        chosen.append(index)
        strata[index] = "random"
    return {"indices": chosen, "strata": strata, "pairs": pairs}


# ---------------------------------------------------------------- planning


def plan_run(
    records: list[dict],
    functional: np.ndarray,
    lineage: np.ndarray,
    probes_path: str | Path,
    *,
    n_agents: int = 60,
    replicates: int = 2,
    model: str = "claude-opus-5-5",
    effort: str | None = "low",
    backend: str = "claude-cli",
    max_tokens: int = 8000,
    judge_model: str = "claude-opus-5-5",
    seed: int = 7,
    runs_dir: Path = RUNS_DIR,
    commits: dict[str, str] | None = None,
) -> Path:
    """Freeze an experiment: sampled agents, settings, genotype distances and every request key."""
    battery = load_probes(probes_path)
    probes = battery["probes"]
    sample = select_agents(records, functional, n_agents, seed)
    indices = sample["indices"]
    if backend not in ("claude-cli", "anthropic-batch"):
        raise ValueError(f"unknown backend {backend}")
    settings = {"backend": backend, "effort": effort, "max_tokens": max_tokens if backend == "anthropic-batch" else None}
    code_hash = sha(Path(__file__).read_text(encoding="utf-8"), 12)
    created = _now()
    run_id = f"{created[:10]}-{sha(json.dumps([settings, model, indices, code_hash, created]), 8)}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    agents = [
        {
            "agent_id": str(records[i]["agent_id"]),
            "source": str(records[i]["source"]),
            "source_path": str(records[i]["source_path"]),
            "source_commit": (commits or {}).get(str(records[i]["source"])),
            "stratum": sample["strata"][i],
            "system_sha": sha(str(records[i]["prompt"]), 16),
            "purpose": str(records[i]["purpose"]),
            "system": str(records[i]["prompt"]),
        }
        for i in indices
    ]
    agents.append({"agent_id": BASELINE_ID, "source": "-", "stratum": "baseline", "system_sha": "", "system": None})
    _append_jsonl(run_dir / "agents.jsonl", agents)

    position = {index: k for k, index in enumerate(indices)}
    genotype = {
        "agent_ids": [a["agent_id"] for a in agents[:-1]],
        "functional": functional[np.ix_(indices, indices)].round(5).tolist(),
        "lineage": lineage[np.ix_(indices, indices)].round(5).tolist(),
        "pairs": {k: [[position[i], position[j]] for i, j in v] for k, v in sample["pairs"].items()},
    }
    (run_dir / "genotype.json").write_text(json.dumps(genotype), encoding="utf-8")

    requests = [
        {
            "key": response_key(model, settings, agent["system"], probe, rep),
            "agent_id": agent["agent_id"],
            "probe_id": probe["id"],
            "probe_hash": probe["hash"],
            "rep": rep,
        }
        for agent in agents
        for probe in probes
        for rep in range(replicates)
    ]
    _append_jsonl(run_dir / "requests.jsonl", requests)

    manifest = {
        "run_id": run_id,
        "created": created,
        "model": model,
        "settings": settings,
        "judge_model": judge_model,
        "judge_version": JUDGE_VERSION,
        "replicates": replicates,
        "seed": seed,
        "code_hash": code_hash,
        "probes_file": str(probes_path),
        "probes_version": battery.get("version"),
        "probes": {p["id"]: p["hash"] for p in probes},
        "n_agents": len(agents),
        "strata": {s: sum(1 for a in agents if a["stratum"] == s) for s in ("twin", "near", "random", "baseline")},
        "n_requests": len(requests),
        "estimate": estimate_cost(agents, probes, replicates, model, judge_model, backend),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def estimate_cost(
    agents: list[dict], probes: list[dict], replicates: int, model: str, judge_model: str, backend: str
) -> dict:
    """Rough cost at list price (claude-cli) or batch price. Input ~ chars/4; output ~1,200 tokens.

    For claude-cli on a subscription this is API-equivalent spend that counts against usage limits.
    """
    prices = LIST_PRICES if backend == "claude-cli" else BATCH_PRICES
    in_price, out_price = prices.get(model, (4.0, 20.0))
    j_in, j_out = prices.get(judge_model, (4.0, 20.0))
    calls = len(agents) * len(probes) * replicates
    input_tokens = sum(
        (len(a["system"] or "") + len(p["prompt"])) / 4 + 50 for a in agents for p in probes
    ) * replicates
    output_tokens = calls * 1200
    judge_in = calls * (1200 + 600)
    judge_out = calls * 400
    respond = (input_tokens * in_price + output_tokens * out_price) / 1e6
    judge = (judge_in * j_in + judge_out * j_out) / 1e6
    return {
        "basis": "list" if backend == "claude-cli" else "batch",
        "calls": calls,
        "respond_usd": round(respond, 2),
        "judge_usd": round(judge, 2),
        "total_usd": round(respond + judge, 2),
    }


# ---------------------------------------------------------------- batches


def _load_run(run_dir: Path) -> tuple[dict, dict[str, dict], dict[str, dict], list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    agents = {a["agent_id"]: a for a in _read_jsonl(run_dir / "agents.jsonl")}
    battery = load_probes(manifest["probes_file"])
    probes = {p["id"]: p for p in battery["probes"]}
    for probe_id, probe_hash in manifest["probes"].items():
        if probes.get(probe_id, {}).get("hash") != probe_hash:
            raise ValueError(f"probe {probe_id} changed since this run was planned; plan a new run")
    return manifest, agents, probes, _read_jsonl(run_dir / "requests.jsonl")


def _store(runs_dir: Path, name: str) -> dict[str, dict]:
    return {row["key"]: row for row in _read_jsonl(runs_dir / "store" / f"{name}.jsonl")}


def _pending_batches(run_dir: Path, phase: str) -> set[str]:
    """Keys already inside a submitted, not-yet-collected batch for this phase."""
    keys: set[str] = set()
    for row in _read_jsonl(run_dir / "batches.jsonl"):
        if row["phase"] == phase and row["event"] == "submitted":
            keys.update(row["keys"])
        if row["phase"] == phase and row["event"] == "collected":
            keys.difference_update(row["keys"])
    return keys


def submit_responses(run_dir: Path, client, runs_dir: Path = RUNS_DIR, chunk: int = 5000) -> dict:
    """Submit every request not already in the store or in flight. Safe to re-run."""
    manifest, agents, probes, requests = _load_run(run_dir)
    done = _store(runs_dir, "responses")
    in_flight = _pending_batches(run_dir, "respond")
    todo = [r for r in requests if r["key"] not in done and r["key"] not in in_flight]
    todo = list({r["key"]: r for r in todo}.values())
    settings = manifest["settings"]
    batch_ids = []
    for start in range(0, len(todo), chunk):
        part = todo[start : start + chunk]
        batch_requests = []
        for row in part:
            params: dict = {
                "model": manifest["model"],
                "max_tokens": settings["max_tokens"],
                "messages": [{"role": "user", "content": probes[row["probe_id"]]["prompt"]}],
            }
            system = agents[row["agent_id"]]["system"]
            if system:
                params["system"] = system
            if settings.get("effort") and not manifest["model"].startswith("claude-haiku"):
                params["output_config"] = {"effort": settings["effort"]}
            batch_requests.append({"custom_id": row["key"], "params": params})
        batch = client.messages.batches.create(requests=batch_requests)
        batch_ids.append(batch.id)
        _append_jsonl(
            run_dir / "batches.jsonl",
            [{"phase": "respond", "event": "submitted", "batch_id": batch.id, "at": _now(), "keys": [r["key"] for r in part]}],
        )
    return {"submitted": len(todo), "reused_from_store": len(requests) - len(todo) - len(in_flight), "batches": batch_ids}


def _text_of(message) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


def collect(run_dir: Path, client, runs_dir: Path = RUNS_DIR) -> dict:
    """Fetch every ended batch into the store. Returns counts of what is still processing."""
    status = {"collected": 0, "still_processing": 0, "errored": 0}
    batches = _read_jsonl(run_dir / "batches.jsonl")
    collected_ids = {b["batch_id"] for b in batches if b["event"] == "collected"}
    for row in batches:
        if row["event"] != "submitted" or row["batch_id"] in collected_ids:
            continue
        batch = client.messages.batches.retrieve(row["batch_id"])
        if batch.processing_status != "ended":
            status["still_processing"] += 1
            continue
        store_name = "responses" if row["phase"] == "respond" else "judgments"
        out, keys = [], []
        for result in client.messages.batches.results(row["batch_id"]):
            keys.append(result.custom_id)
            entry = {"key": result.custom_id, "batch_id": row["batch_id"], "collected_at": _now(), "result_type": result.result.type}
            if result.result.type == "succeeded":
                message = result.result.message
                entry.update(
                    {
                        "model": message.model,
                        "message_id": message.id,
                        "stop_reason": message.stop_reason,
                        "text": _text_of(message),
                        "usage": {"input": message.usage.input_tokens, "output": message.usage.output_tokens},
                    }
                )
                out.append(entry)
            else:
                status["errored"] += 1
                keys.pop()  # leave it un-collected so a later submit retries it
        _append_jsonl(runs_dir / "store" / f"{store_name}.jsonl", out)
        _append_jsonl(run_dir / "batches.jsonl", [{"phase": row["phase"], "event": "collected", "batch_id": row["batch_id"], "at": _now(), "keys": keys}])
        status["collected"] += len(out)
    return status


def _judge_prompt(probe: dict, answer: str) -> str:
    planted = probe.get("planted") or "none (answer n/a)"
    return (
        "You are grading one assistant response for a behavioural study. Judge only what the response does.\n\n"
        f"<task>\n{probe['prompt']}\n</task>\n\n<planted_issue>\n{planted}\n</planted_issue>\n\n"
        f"<response>\n{answer}\n</response>\n\n"
        "Fields: asked_clarifying_question (asks the user something before or instead of answering); "
        "declined_or_redirected (declines, or says it is outside its role); pushed_back (disagrees with the "
        "user's framing or request); wrote_code; wrote_tests; flagged_security (raises a security concern); "
        "caught_planted_issue (did it identify the planted issue: yes / partly / no, or n/a when there is none); "
        "approach (the response's approach in at most 20 words)."
    )


def submit_judgments(run_dir: Path, client, runs_dir: Path = RUNS_DIR, chunk: int = 5000) -> dict:
    """Grade every collected response that has no judgment yet."""
    manifest, _, probes, requests = _load_run(run_dir)
    responses = _store(runs_dir, "responses")
    judged = _store(runs_dir, "judgments")
    in_flight = _pending_batches(run_dir, "judge")
    judge_model = manifest["judge_model"]
    todo = {}
    for row in requests:
        response = responses.get(row["key"])
        key = judge_key(judge_model, row["key"])
        if response and key not in judged and key not in in_flight:
            todo[key] = (row, response)
    items = list(todo.items())
    batch_ids = []
    for start in range(0, len(items), chunk):
        part = items[start : start + chunk]
        batch = client.messages.batches.create(
            requests=[
                {
                    "custom_id": key,
                    "params": {
                        "model": judge_model,
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": _judge_prompt(probes[row["probe_id"]], response["text"])}],
                        "output_config": {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                    },
                }
                for key, (row, response) in part
            ]
        )
        batch_ids.append(batch.id)
        _append_jsonl(
            run_dir / "batches.jsonl",
            [{"phase": "judge", "event": "submitted", "batch_id": batch.id, "at": _now(), "keys": [k for k, _ in part]}],
        )
    return {"submitted": len(items), "batches": batch_ids}


# ---------------------------------------------------------------- local Claude Code backend


def claude_cli_call(
    system: str,
    prompt: str,
    model: str,
    effort: str | None,
    schema: dict | None = None,
    timeout: int = 600,
    claude_bin: str | None = None,
) -> dict:
    """One headless Claude Code call, sandboxed: no tools, no settings/hooks/MCP, empty temp cwd."""
    binary = claude_bin or shutil.which("claude") or "claude"
    cmd = [
        binary, "-p",
        "--output-format", "json",
        "--no-session-persistence",
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--system-prompt", system,
        "--model", model,
    ]
    if effort:
        cmd += ["--effort", effort]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    with tempfile.TemporaryDirectory(prefix="phailogeny-sbx-") as sandbox:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=sandbox, timeout=timeout)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:], "exit_code": proc.returncode}
    usage = payload.get("usage") or {}
    models = list((payload.get("modelUsage") or {}).keys())
    return {
        "ok": not payload.get("is_error") and payload.get("subtype") == "success",
        "text": payload.get("result") or "",
        "structured": payload.get("structured_output"),
        "stop_reason": payload.get("stop_reason"),
        "session_id": payload.get("session_id"),
        "model": models[0] if models else model,
        "cost_usd": payload.get("total_cost_usd"),
        "duration_ms": payload.get("duration_ms"),
        "usage": {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "thinking": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        },
        "error": None if not payload.get("is_error") else str(payload.get("result"))[:2000],
    }


def run_cli(
    run_dir: Path,
    phase: str = "respond",
    workers: int = 6,
    limit: int | None = None,
    runs_dir: Path = RUNS_DIR,
    call=claude_cli_call,
) -> dict:
    """Run missing responses (or judgments) through local Claude Code, `workers` at a time.

    Each result is appended to the shared store the moment it arrives, so an interrupted run
    resumes where it stopped. Failures are logged to the run's errors.jsonl and retried next time.
    """
    manifest, agents, probes, requests = _load_run(run_dir)
    if manifest["settings"].get("backend") != "claude-cli":
        raise ValueError("this run was planned for the anthropic-batch backend")
    model, effort = manifest["model"], manifest["settings"]["effort"]
    responses = _store(runs_dir, "responses")
    jobs: dict[str, tuple] = {}
    if phase == "respond":
        for row in requests:
            if row["key"] not in responses:
                system = agents[row["agent_id"]]["system"] or BASELINE_SYSTEM
                jobs[row["key"]] = (row, system, probes[row["probe_id"]]["prompt"], None, model)
        store_name = "responses"
    else:
        judged = _store(runs_dir, "judgments")
        judge_model = manifest["judge_model"]
        for row in requests:
            key = judge_key(judge_model, row["key"])
            if row["key"] in responses and key not in judged:
                prompt = _judge_prompt(probes[row["probe_id"]], responses[row["key"]]["text"])
                jobs[key] = (row, "You grade assistant responses precisely and briefly.", prompt, JUDGE_SCHEMA, judge_model)
        store_name = "judgments"
    items = list(jobs.items())[: limit or None]
    lock = threading.Lock()
    status = {"phase": phase, "queued": len(items), "done": 0, "failed": 0, "cost_usd": 0.0}

    def work(key: str, job: tuple) -> tuple[str, tuple, dict]:
        row, system, prompt, schema, job_model = job
        try:
            return key, job, call(system, prompt, job_model, effort if phase == "respond" else "low", schema)
        except Exception as exc:  # timeouts, missing binary
            return key, job, {"ok": False, "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(work, key, job) for key, job in items):
            key, job, result = future.result()
            row = job[0]
            with lock:
                if result.get("ok"):
                    entry = {
                        "key": key,
                        "backend": "claude-cli",
                        "run_id": manifest["run_id"],
                        "agent_id": row["agent_id"],
                        "probe_id": row["probe_id"],
                        "rep": row["rep"],
                        "collected_at": _now(),
                        **{k: result.get(k) for k in ("model", "stop_reason", "session_id", "cost_usd", "duration_ms", "usage")},
                        "text": json.dumps(result["structured"]) if phase == "judge" else result["text"],
                    }
                    _append_jsonl(runs_dir / "store" / f"{store_name}.jsonl", [entry])
                    status["done"] += 1
                    status["cost_usd"] += result.get("cost_usd") or 0.0
                else:
                    _append_jsonl(run_dir / "errors.jsonl", [{"key": key, "phase": phase, "at": _now(), "error": result.get("error")}])
                    status["failed"] += 1
    status["cost_usd"] = round(status["cost_usd"], 4)
    return status


# ---------------------------------------------------------------- analysis


def structure_features(text: str) -> list[float]:
    """Style fingerprint: size and formatting, independent of content."""
    lines = text.splitlines()
    return [
        float(np.log1p(len(text.split()))),
        float(sum(1 for l in lines if l.lstrip().startswith("#"))),
        float(text.count("```") // 2),
        float(sum(1 for l in lines if l.lstrip().startswith(("- ", "* ")))),
        float(sum(1 for l in lines if l.lstrip()[:3].rstrip(".").isdigit() and l.lstrip()[1:3].startswith((".", ")")))),
        float(text.count("?")),
    ]


def mantel(a: np.ndarray, b: np.ndarray, permutations: int = 999, seed: int = 0) -> dict:
    """Spearman Mantel test between two distance matrices (joint row/column permutation)."""
    from scipy.stats import rankdata

    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    x = rankdata(a[iu])
    rng = np.random.default_rng(seed)

    def corr(matrix: np.ndarray) -> float:
        y = rankdata(matrix[iu])
        return float(np.corrcoef(x, y)[0, 1])

    observed = corr(b)
    hits = sum(corr(b[np.ix_(p, p)]) >= observed for p in (rng.permutation(n) for _ in range(permutations)))
    return {"r": round(observed, 3), "p": round((hits + 1) / (permutations + 1), 4), "n_agents": n}


def analyse_run(run_dir: Path, runs_dir: Path = RUNS_DIR, model_name: str = "all-MiniLM-L6-v2") -> dict:
    """Behaviour distances vs genotype distances, twins vs random, noise ceiling, baseline."""
    from sentence_transformers import SentenceTransformer

    manifest, agents, probes, requests = _load_run(run_dir)
    genotype = json.loads((run_dir / "genotype.json").read_text(encoding="utf-8"))
    responses = _store(runs_dir, "responses")
    judgments = _store(runs_dir, "judgments")
    ids = genotype["agent_ids"] + [BASELINE_ID]
    probe_ids = list(manifest["probes"])
    reps = manifest["replicates"]
    index = {a: k for k, a in enumerate(ids)}
    pindex = {p: k for k, p in enumerate(probe_ids)}

    have = [r for r in requests if r["key"] in responses]
    coverage = len(have) / len(requests)
    model = SentenceTransformer(model_name)
    vectors = model.encode([responses[r["key"]]["text"] or " " for r in have], normalize_embeddings=True, show_progress_bar=False)
    dim = vectors.shape[1]
    emb = np.full((len(ids), len(probe_ids), reps, dim), np.nan)
    style = np.full((len(ids), len(probe_ids), reps, 6), np.nan)
    disp = np.full((len(ids), len(probe_ids), reps, len(JUDGE_FIELDS)), np.nan)
    caught: dict[str, list[float]] = {}
    for row, vec in zip(have, vectors):
        a, p, r = index[row["agent_id"]], pindex[row["probe_id"]], row["rep"]
        emb[a, p, r] = vec
        style[a, p, r] = structure_features(responses[row["key"]]["text"])
        verdict = judgments.get(judge_key(manifest["judge_model"], row["key"]))
        if verdict and verdict.get("text"):
            parsed = json.loads(verdict["text"])
            disp[a, p, r] = [float(parsed[f]) for f in JUDGE_FIELDS]
            if parsed["caught_planted_issue"] != "n/a":
                caught.setdefault(row["agent_id"], []).append({"yes": 1.0, "partly": 0.5, "no": 0.0}[parsed["caught_planted_issue"]])

    # Content: per probe, cosine distance between replicate-mean embeddings, averaged over probes.
    mean_emb = np.nanmean(emb, axis=2)
    mean_emb /= np.linalg.norm(mean_emb, axis=2, keepdims=True)
    content = np.nanmean(1.0 - np.einsum("apd,bpd->abp", mean_emb, mean_emb), axis=2)
    # Noise ceiling: same agent, same probe, different replicate.
    noise = float(np.nanmean(1.0 - np.einsum("apd,apd->ap", emb[:, :, 0], emb[:, :, 1]))) if reps > 1 else None

    disp_profile = np.nanmean(disp, axis=(1, 2)) if np.isfinite(disp).any() else None
    disposition = np.abs(disp_profile[:, None, :] - disp_profile[None, :, :]).mean(axis=2) if disp_profile is not None else None
    style_profile = np.nanmean(style, axis=(1, 2))
    style_z = (style_profile - style_profile.mean(0)) / (style_profile.std(0) + 1e-9)
    structure = np.linalg.norm(style_z[:, None, :] - style_z[None, :, :], axis=2)

    k = len(genotype["agent_ids"])
    functional = np.array(genotype["functional"])
    lineage_d = 1.0 - np.array(genotype["lineage"])
    tests = {
        "content_vs_functional": mantel(functional, content[:k, :k]),
        "structure_vs_functional": mantel(functional, structure[:k, :k]),
        "content_vs_structure": mantel(structure[:k, :k], content[:k, :k]),
    }
    if disposition is not None:
        tests["disposition_vs_functional"] = mantel(functional, disposition[:k, :k])

    def pair_mean(matrix: np.ndarray, pairs: list[list[int]]) -> float | None:
        return round(float(np.mean([matrix[i, j] for i, j in pairs])), 4) if pairs else None

    iu = np.triu_indices(k, k=1)
    comparisons = {
        "content_distance": {
            "twin_pairs": pair_mean(content, genotype["pairs"].get("twin", [])),
            "near_pairs": pair_mean(content, genotype["pairs"].get("near", [])),
            "all_pairs": round(float(content[:k, :k][iu].mean()), 4),
            "same_agent_replicates": round(noise, 4) if noise is not None else None,
        },
        "functional_distance": {
            "twin_pairs": pair_mean(functional, genotype["pairs"].get("twin", [])),
            "all_pairs": round(float(functional[iu].mean()), 4),
        },
        "lineage_distance": {"twin_pairs": pair_mean(lineage_d, genotype["pairs"].get("twin", []))},
    }
    baseline = content[:k, k]
    per_agent = [
        {
            "agent_id": ids[i],
            "stratum": agents[ids[i]]["stratum"],
            "distance_from_baseline": round(float(baseline[i]), 4),
            "caught_planted_rate": round(float(np.mean(caught[ids[i]])), 3) if ids[i] in caught else None,
        }
        for i in range(k)
    ]
    usage = [responses[r["key"]].get("usage", {}) for r in have]
    analysis = {
        "run_id": manifest["run_id"],
        "analysed_at": _now(),
        "coverage": round(coverage, 4),
        "judged": round(float(np.isfinite(disp[..., 0]).mean()), 4),
        "mantel": tests,
        "comparisons": comparisons,
        "baseline": {
            "mean_distance_from_baseline": round(float(baseline.mean()), 4),
            "baseline_caught_planted_rate": round(float(np.mean(caught[BASELINE_ID])), 3) if BASELINE_ID in caught else None,
        },
        "truncated_responses": sum(1 for r in have if responses[r["key"]].get("stop_reason") == "max_tokens"),
        "tokens": {"input": sum(u.get("input", 0) for u in usage), "output": sum(u.get("output", 0) for u in usage)},
        "per_agent": per_agent,
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "phenotype_matrices.npz", ids=np.array(ids), content=content, structure=structure,
                        disposition=disposition if disposition is not None else np.zeros(0))
    (run_dir / "report.md").write_text(render_report(manifest, analysis), encoding="utf-8")
    _append_jsonl(
        runs_dir / "index.jsonl",
        [{
            "run_id": manifest["run_id"],
            "analysed_at": analysis["analysed_at"],
            "model": manifest["model"],
            "settings": manifest["settings"],
            "judge_model": manifest["judge_model"],
            "probes_version": manifest["probes_version"],
            "code_hash": manifest["code_hash"],
            "n_agents": manifest["n_agents"],
            "coverage": analysis["coverage"],
            "mantel_content_r": tests["content_vs_functional"]["r"],
            "mantel_content_p": tests["content_vs_functional"]["p"],
            "twin_content_distance": comparisons["content_distance"]["twin_pairs"],
            "all_content_distance": comparisons["content_distance"]["all_pairs"],
            "noise_ceiling": comparisons["content_distance"]["same_agent_replicates"],
        }],
    )
    return analysis


def render_report(manifest: dict, analysis: dict) -> str:
    """Human-readable summary of one run."""
    lines = [
        f"# Common-garden run {manifest['run_id']}",
        "",
        f"- Model: `{manifest['model']}` {manifest['settings']}; judge `{manifest['judge_model']}` v{manifest['judge_version']}",
        f"- Agents: {manifest['n_agents']} {manifest['strata']}; probes v{manifest['probes_version']} ({len(manifest['probes'])}); replicates {manifest['replicates']}",
        f"- Coverage: {analysis['coverage']:.0%} responses, {analysis['judged']:.0%} judged; truncated: {analysis['truncated_responses']}",
        f"- Code hash `{manifest['code_hash']}`; tokens in/out {analysis['tokens']['input']:,} / {analysis['tokens']['output']:,}",
        "",
        "## Does behaviour track genotype? (Spearman Mantel, 999 permutations)",
        "",
        "| Test | r | p |",
        "| --- | --- | --- |",
        *[f"| {name} | {t['r']} | {t['p']} |" for name, t in analysis["mantel"].items()],
        "",
        "## Content distance between responses",
        "",
        "| Pairs | Mean distance |",
        "| --- | --- |",
        *[f"| {name} | {value} |" for name, value in analysis["comparisons"]["content_distance"].items()],
        "",
        "Twins below all pairs = same-named agents behave alike. The replicate row is the noise floor:",
        "no pair of different agents can be expected to score below it.",
        "",
        f"Mean distance from the no-system-prompt baseline: {analysis['baseline']['mean_distance_from_baseline']}",
    ]
    return "\n".join(lines) + "\n"


def wait_and_collect(run_dir: Path, client, runs_dir: Path = RUNS_DIR, poll_seconds: int = 60) -> dict:
    """Poll until every submitted batch has been collected."""
    while True:
        status = collect(run_dir, client, runs_dir)
        if status["still_processing"] == 0:
            return status
        time.sleep(poll_seconds)
