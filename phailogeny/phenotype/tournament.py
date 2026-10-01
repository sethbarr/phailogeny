"""Clade tournament, preparation stage: claim matrix, task generation, local task validation.

A clade (a branch of the tree) holds agents that claim similar jobs. To rank them fairly:
1. freeze the clade's members;
2. have a model read every member's description and list the capabilities the clade claims
   (the claim matrix), so tests come from the clade's consensus, not one agent's wording;
3. generate tasks per capability, mixing verifiable (planted bug + test), rubric, disposition
   and robustness tasks;
4. validate verifiable tasks locally: the buggy code must fail its test, the reference fix pass.
Hold-out tasks from an independent source (data/probes.yaml) are added unchanged.

Everything is written for human review before any agent runs. Model calls go through local,
sandboxed Claude Code (`claude_cli_call`), and every call's output and cost is logged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from phailogeny.phenotype.common_garden import claude_cli_call, load_probes, sha

GEN_VERSION = 1
TASK_TYPES = ["verifiable", "rubric", "disposition", "robustness"]
SYSTEM = "You design evaluation tasks for a research study comparing AI agents. Be concrete and precise."

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}, "definition": {"type": "string"}},
                "required": ["id", "name", "definition"],
                "additionalProperties": False,
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"agent_id": {"type": "string"}, "capability_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["agent_id", "capability_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["capabilities", "claims"],
    "additionalProperties": False,
}

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": TASK_TYPES},
                    "title": {"type": "string"},
                    "prompt": {"type": "string"},
                    "buggy_code": {"type": "string"},
                    "test_code": {"type": "string"},
                    "reference_code": {"type": "string"},
                    "rubric": {"type": "array", "items": {"type": "string"}},
                    "expected_behaviour": {"type": "string"},
                    "variant_of": {"type": "string"},
                },
                "required": ["id", "type", "title", "prompt", "buggy_code", "test_code", "reference_code",
                             "rubric", "expected_behaviour", "variant_of"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def freeze_members(tree: object, anchor: str, records: list[dict], min_tips: int = 18, max_tips: int = 40) -> list[dict]:
    """The clade around `anchor`, with each member's source, path and prompt hash."""
    from phailogeny.report.figures import clade_around

    by_id = {str(r["agent_id"]): r for r in records}
    clade = clade_around(tree, anchor, min_tips, max_tips)
    members = []
    for tip in clade.tips():
        record = by_id[str(tip.name)]
        members.append({
            "agent_id": str(tip.name),
            "source": str(record["source"]),
            "source_path": str(record["source_path"]),
            "prompt_sha": sha(f"{record['purpose']}\n{record['prompt']}", 16),
        })
    return members


def build_claim_matrix(members: list[dict], records: dict[str, dict], call=claude_cli_call, model: str = "claude-opus-5-5") -> tuple[dict, dict]:
    """Ask a model which capabilities the clade claims, and which member claims which."""
    blocks = []
    for member in members:
        record = records[member["agent_id"]]
        excerpt = " ".join(str(record["prompt"]).split()[:250])
        blocks.append(f"<agent id=\"{member['agent_id']}\">\n<description>{record['purpose']}</description>\n<prompt_excerpt>{excerpt}</prompt_excerpt>\n</agent>")
    prompt = (
        "Below are AI agents that a similarity analysis grouped together. List the distinct, testable "
        "capabilities this group claims (6 to 9; short snake_case ids; one-sentence definitions that say what "
        "a good performance looks like). Then, for every agent, list the capability ids its description or "
        "prompt claims. Use only what the text says; do not infer.\n\n" + "\n\n".join(blocks)
    )
    result = call(SYSTEM, prompt, model, "medium", CLAIM_SCHEMA)
    return result.get("structured") or {}, result


def generate_tasks(capability: dict, clade_summary: str, n: int, call=claude_cli_call, model: str = "claude-opus-5-5") -> tuple[list[dict], dict]:
    """Tasks testing one capability, written from the clade's consensus definition."""
    prompt = f"""Write {n} evaluation tasks for this capability of AI debugging agents.

Capability: {capability['name']} ({capability['id']})
Definition: {capability['definition']}
The agents being compared: {clade_summary}

Mix of types (adjust only if the capability cannot support one):
- About half "verifiable": a self-contained Python 3 module (standard library only, under 60 lines) with ONE
  planted bug in `buggy_code`; `test_code` = a pytest file that does `from buggy import *` and has test_
  functions that FAIL on the buggy code and PASS on the fix; `reference_code` = the full fixed module.
  The `prompt` shown to the agent includes the buggy code and a realistic symptom (error output or wrong
  result) but NOT the test or the fix, and asks for the full corrected module in one ```python block.
- "rubric": an open-ended task (logs, traces, an incident description); `rubric` = 3-5 checkable criteria
  written now, before any answer exists.
- "disposition": a request where the right behaviour is to ask, decline, or push back;
  `expected_behaviour` = what a good agent does.
- "robustness": a reworded, noisier variant of one of your other tasks; `variant_of` = that task's id.

Rules: every prompt is self-contained (no repo, no tools); realistic; ids like "{capability['id']}_1".
Unused fields are empty strings or empty lists."""
    result = call(SYSTEM, prompt, model, "medium", TASK_SCHEMA)
    tasks = (result.get("structured") or {}).get("tasks", [])
    for task in tasks:
        task["capability"] = capability["id"]
    return tasks, result


def _sandbox_profile(workdir: str) -> str:
    """macOS sandbox: no network, writes only inside `workdir`, no reading the user's files
    except the Python runtime and `workdir`."""
    home = str(Path.home())
    python_dirs = {sys.prefix, sys.base_prefix}
    reads = " ".join(f'(subpath "{d}")' for d in sorted(python_dirs | {workdir}))
    return (
        "(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
        f'(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/tty"))\n'
        f'(deny file-read-data (subpath "{home}"))\n'
        f"(allow file-read-data {reads})\n"
    )


def run_tests(module_code: str, test_code: str, timeout: int = 30) -> dict:
    """Run a task's pytest file against one version of the module, in a throwaway directory.

    On macOS the run is wrapped in `sandbox-exec` (no network, no writes outside the directory,
    no reading the user's files). Elsewhere it falls back to a stripped environment and a timeout
    only, and says so in the result.
    """
    with tempfile.TemporaryDirectory(prefix="phailogeny-tests-") as tmp:
        tmp = os.path.realpath(tmp)
        Path(tmp, "buggy.py").write_text(module_code, encoding="utf-8")
        Path(tmp, "test_buggy.py").write_text(test_code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_buggy.py"]
        sandboxed = sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()
        if sandboxed:
            Path(tmp, "profile.sb").write_text(_sandbox_profile(tmp), encoding="utf-8")
            cmd = ["/usr/bin/sandbox-exec", "-f", str(Path(tmp, "profile.sb")), *cmd]
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": False, "timeout": True, "output": "", "sandboxed": sandboxed}
    return {"passed": proc.returncode == 0, "timeout": False, "output": proc.stdout[-1500:], "sandboxed": sandboxed}


def validate_task(task: dict) -> dict:
    """Verifiable tasks must fail on the buggy code and pass on the reference fix."""
    if task["type"] != "verifiable":
        missing = []
        if task["type"] == "rubric" and len(task["rubric"]) < 3:
            missing.append("rubric needs 3+ criteria")
        if task["type"] == "disposition" and not task["expected_behaviour"]:
            missing.append("no expected behaviour")
        if task["type"] == "robustness" and not task["variant_of"]:
            missing.append("no variant_of")
        return {"valid": not missing, "reasons": missing}
    if not (task["buggy_code"] and task["test_code"] and task["reference_code"]):
        return {"valid": False, "reasons": ["missing code, test or reference"]}
    if task["buggy_code"].strip() not in task["prompt"] and "```" not in task["prompt"]:
        return {"valid": False, "reasons": ["prompt does not show the code"]}
    buggy = run_tests(task["buggy_code"], task["test_code"])
    fixed = run_tests(task["reference_code"], task["test_code"])
    reasons = []
    if buggy["passed"]:
        reasons.append("test passes on buggy code (bug not caught)")
    if not fixed["passed"]:
        reasons.append("test fails on reference fix")
    return {"valid": not reasons, "reasons": reasons, "buggy_output": buggy["output"][-400:], "fixed_output": fixed["output"][-400:]}


def holdout_tasks(probes_path: str | Path, probe_ids: list[str]) -> list[dict]:
    """Independent tasks (not derived from any agent's description) for a generalisation check."""
    battery = load_probes(probes_path)
    out = []
    for probe in battery["probes"]:
        if probe["id"] in probe_ids:
            out.append({
                "id": f"holdout_{probe['id']}", "type": "holdout", "capability": "holdout", "title": probe["id"],
                "prompt": probe["prompt"], "buggy_code": "", "test_code": "", "reference_code": "",
                "rubric": [f"Identifies: {probe['planted']}"] if probe.get("planted") else [],
                "expected_behaviour": "", "variant_of": "", "validation": {"valid": True, "reasons": []},
            })
    return out


def prepare(
    tree: object,
    anchor: str,
    records: list[dict],
    out_dir: Path,
    log_path: Path,
    tasks_per_capability: int = 6,
    workers: int = 6,
    holdout_ids: list[str] | None = None,
    call=claude_cli_call,
) -> dict:
    """Freeze members, build the claim matrix, generate and validate tasks; write files for review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(r["agent_id"]): r for r in records}
    members = freeze_members(tree, anchor, records)
    (out_dir / "members.json").write_text(json.dumps({"anchor": anchor, "frozen_at": _now(), "members": members}, indent=2), encoding="utf-8")

    claims, raw = build_claim_matrix(members, by_id, call)
    _log(log_path, {"step": "claims", "gen_version": GEN_VERSION, "at": _now(), "cost_usd": raw.get("cost_usd"), "ok": raw.get("ok"), "output": claims})
    if not claims.get("capabilities"):
        raise RuntimeError(f"claim matrix failed: {raw.get('error')}")
    (out_dir / "claims.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")

    summary = "; ".join(f"{m['agent_id']}" for m in members)
    tasks: list[dict] = []
    cost = raw.get("cost_usd") or 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_tasks, cap, summary, tasks_per_capability, call) for cap in claims["capabilities"]]
        for cap, future in zip(claims["capabilities"], futures):
            generated, result = future.result()
            cost += result.get("cost_usd") or 0.0
            _log(log_path, {"step": "tasks", "capability": cap["id"], "gen_version": GEN_VERSION, "at": _now(),
                            "cost_usd": result.get("cost_usd"), "ok": result.get("ok"), "output": generated})
            tasks.extend(generated)
    for task in tasks:
        task["validation"] = validate_task(task)
    tasks.extend(holdout_tasks("data/probes.yaml", holdout_ids or ["A2_debug", "B3_ci", "E4_pushback"]))

    (out_dir / "tasks.yaml").write_text(
        yaml.safe_dump({"gen_version": GEN_VERSION, "anchor": anchor, "tasks": tasks}, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    by_type: dict[str, dict[str, int]] = {}
    for task in tasks:
        row = by_type.setdefault(task["type"], {"total": 0, "valid": 0})
        row["total"] += 1
        row["valid"] += int(task["validation"]["valid"])
    report = {
        "members": len(members),
        "capabilities": [c["id"] for c in claims["capabilities"]],
        "tasks_by_type": by_type,
        "invalid": [{"id": t["id"], "reasons": t["validation"]["reasons"]} for t in tasks if not t["validation"]["valid"]],
        "generation_cost_usd": round(cost, 3),
    }
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------- run and rank (verifiable tasks)

BASELINE_SYSTEM = "You are a helpful assistant."
GENERIC_SYSTEM = (
    "You are an expert software engineer. When given a bug, find the root cause, make the smallest fix "
    "that removes it, and check the fix against the reported symptoms."
)
CONTROLS = {"control/no-prompt": BASELINE_SYSTEM, "control/generic-expert": GENERIC_SYSTEM}


def _code_blocks(text: str) -> list[tuple[str, str]]:
    blocks, inside, current, lang = [], False, [], ""
    for line in text.splitlines():
        stripped = line.strip()
        fence = stripped[:3] in ("```", "~~~")
        if fence and inside and not stripped.strip("`~"):
            blocks.append((lang, "\n".join(current)))
            inside, current = False, []
            continue
        if fence and not inside:
            inside, lang = True, stripped.lstrip("`~").strip().lower()
            continue
        if inside:
            current.append(line)
    return blocks


def _top_level_names(code: str) -> set[str]:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def extract_code(text: str, reference_code: str = "") -> str | None:
    """The answer's corrected module: the Python block that defines most of the reference module's
    top-level names (so a test block the agent adds after its fix is not graded instead).
    Ties go to the later block. Without a reference, the last Python block."""
    blocks = [code for lang, code in _code_blocks(text) if lang in ("python", "py", "")] or [c for _, c in _code_blocks(text)]
    if not blocks:
        return None
    wanted = _top_level_names(reference_code)
    if not wanted:
        return blocks[-1]
    scored = [(len(wanted & _top_level_names(code)), k) for k, code in enumerate(blocks)]
    return blocks[max(scored)[1]]


def task_hash(task: dict) -> str:
    return sha(f"{task['prompt']}\n{task['test_code']}", 16)


def run_key(model: str, effort: str, system: str, task: dict, rep: int) -> str:
    return sha(f"{model}|{effort}|{sha(system)}|{task['id']}|{task_hash(task)}|{rep}", 32)


def run_verifiable(
    name: str,
    records: list[dict],
    replicates: int = 2,
    workers: int = 8,
    model: str = "claude-opus-5-5",
    effort: str = "low",
    limit: int | None = None,
    call=claude_cli_call,
    data_root: Path = Path("data/tournaments"),
    runs_root: Path = Path("runs/tournaments"),
) -> dict:
    """Every member and control answers every valid verifiable task; answers are scored by tests.

    Results append to runs/tournaments/<name>/results.jsonl keyed by model, effort, system-prompt
    hash, task id + hash and replicate, so an interrupted run resumes and nothing is paid twice.
    """
    import threading
    from concurrent.futures import as_completed

    members = json.loads((data_root / name / "members.json").read_text(encoding="utf-8"))["members"]
    tasks = [t for t in yaml.safe_load((data_root / name / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]
             if t["type"] == "verifiable" and t["validation"]["valid"]]
    by_id = {str(r["agent_id"]): r for r in records}
    systems = {m["agent_id"]: str(by_id[m["agent_id"]]["prompt"]) for m in members} | CONTROLS
    out = runs_root / name / "results.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    jobs = [
        (agent, system, task, rep, run_key(model, effort, system, task, rep))
        for rep in range(replicates) for task in tasks for agent, system in systems.items()
    ]
    jobs = [job for job in jobs if job[4] not in done][: limit or None]
    lock = threading.Lock()
    status = {"queued": len(jobs), "done": 0, "failed_calls": 0, "passed": 0, "cost_usd": 0.0}

    def work(job: tuple) -> tuple[tuple, dict, dict | None]:
        agent, system, task, rep, key = job
        answer = call(system, task["prompt"], model, effort)
        if not answer.get("ok"):
            return job, answer, None
        code = extract_code(answer.get("text") or "", task["reference_code"])
        tested = run_tests(code, task["test_code"]) if code else {"passed": False, "output": "no code block", "sandboxed": None}
        return job, answer, tested

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(work, job) for job in jobs):
            (agent, system, task, rep, key), answer, tested = future.result()
            with lock:
                if tested is None:
                    status["failed_calls"] += 1
                    _log(runs_root / name / "errors.jsonl", {"key": key, "agent": agent, "task": task["id"], "at": _now(), "error": answer.get("error")})
                    continue
                _log(out, {
                    "key": key, "agent_id": agent, "task_id": task["id"], "capability": task["capability"], "rep": rep,
                    "model": answer.get("model"), "effort": effort, "passed": tested["passed"],
                    "code_found": tested["output"] != "no code block", "sandboxed": tested.get("sandboxed"),
                    "test_output": tested["output"][-600:], "cost_usd": answer.get("cost_usd"),
                    "usage": answer.get("usage"), "duration_ms": answer.get("duration_ms"),
                    "session_id": answer.get("session_id"), "at": _now(), "answer": answer.get("text"),
                })
                status["done"] += 1
                status["passed"] += int(tested["passed"])
                status["cost_usd"] += answer.get("cost_usd") or 0.0
    status["cost_usd"] = round(status["cost_usd"], 3)
    return status


def rank(name: str, runs_root: Path = Path("runs/tournaments"), data_root: Path = Path("data/tournaments"),
         n_boot: int = 2000, seed: int = 0) -> dict:
    """Pass rate per agent (mean over tasks of the replicate mean) with bootstrap CIs over tasks."""
    import numpy as np

    rows = [json.loads(line) for line in (runs_root / name / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    agents = sorted({r["agent_id"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})
    capability = {r["task_id"]: r["capability"] for r in rows}
    score = np.full((len(agents), len(tasks)), np.nan)
    cost = {a: [] for a in agents}
    cost_cell = np.full((len(agents), len(tasks)), np.nan)
    cells: dict[tuple[int, int], list[float]] = {}
    costs: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        key = (agents.index(r["agent_id"]), tasks.index(r["task_id"]))
        cells.setdefault(key, []).append(float(r["passed"]))
        costs.setdefault(key, []).append(r.get("cost_usd") or 0.0)
        cost[r["agent_id"]].append(r.get("cost_usd") or 0.0)
    for (i, j), values in cells.items():
        score[i, j] = float(np.mean(values))
        cost_cell[i, j] = float(np.mean(costs[(i, j)]))
    complete = ~np.isnan(score).any(axis=0)
    score = score[:, complete]
    cost_cell = cost_cell[:, complete]
    tasks = [t for t, keep in zip(tasks, complete) if keep]
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(tasks), len(tasks)) for _ in range(n_boot)]
    boots = np.array([score[:, d].mean(axis=1) for d in draws])
    cost_boots = np.array([cost_cell[:, d].mean(axis=1) for d in draws])
    means = score.mean(axis=1)
    from scipy.stats import rankdata

    rank_boot = np.array([rankdata(-b, method="min") for b in boots])  # ties share the best rank
    caps = sorted(set(capability[t] for t in tasks))
    table = []
    for i, agent in enumerate(agents):
        table.append({
            "agent_id": agent,
            "pass_rate": round(float(means[i]), 3),
            "ci95": [round(float(np.percentile(boots[:, i], 2.5)), 3), round(float(np.percentile(boots[:, i], 97.5)), 3)],
            "rank_ci95": [int(np.percentile(rank_boot[:, i], 2.5)), int(np.percentile(rank_boot[:, i], 97.5))],
            "mean_cost_usd": round(float(cost_cell[i].mean()), 4),
            "cost_ci95": [round(float(np.percentile(cost_boots[:, i], 2.5)), 4), round(float(np.percentile(cost_boots[:, i], 97.5)), 4)],
            "by_capability": {c: round(float(score[i, [k for k, t in enumerate(tasks) if capability[t] == c]].mean()), 3) for c in caps},
        })
    table.sort(key=lambda row: -row["pass_rate"])
    result = {
        "n_tasks": len(tasks), "n_agents": len(agents), "replicates": max(r["rep"] for r in rows) + 1,
        "n_results": len(rows), "total_cost_usd": round(sum(sum(v) for v in cost.values()), 2),
        "ranking": table,
        "task_difficulty": {t: round(float(score[:, k].mean()), 3) for k, t in enumerate(tasks)},
    }
    (data_root / name / "ranking.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def rescore(name: str, runs_root: Path = Path("runs/tournaments"), data_root: Path = Path("data/tournaments"), scorer_version: int = 2) -> dict:
    """Re-extract and re-test every stored answer with the current scorer (no model calls).

    Each row keeps its earlier verdict as `passed_v<old>` so the change is traceable.
    """
    tasks = {t["id"]: t for t in yaml.safe_load((data_root / name / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    path = runs_root / name / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row.get("scorer_version", 1) >= scorer_version or not row.get("answer"):
            continue
        task = tasks[row["task_id"]]
        code = extract_code(row["answer"], task["reference_code"])
        tested = run_tests(code, task["test_code"]) if code else {"passed": False, "output": "no code block"}
        row[f"passed_v{row.get('scorer_version', 1)}"] = row["passed"]
        changed += int(tested["passed"] != row["passed"])
        row.update({"passed": tested["passed"], "test_output": tested["output"][-600:], "scorer_version": scorer_version})
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return {"rows": len(rows), "verdicts_changed": changed, "passed": sum(r["passed"] for r in rows)}
