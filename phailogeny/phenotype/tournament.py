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


def run_tests(module_code: str, test_code: str, timeout: int = 30) -> dict:
    """Run a task's pytest file against one version of the module, in a throwaway directory.

    Minimal environment (no inherited credentials), isolated Python, hard timeout. The code was
    written by a model for this study; this is containment, not a security sandbox.
    """
    with tempfile.TemporaryDirectory(prefix="phailogeny-tests-") as tmp:
        Path(tmp, "buggy.py").write_text(module_code, encoding="utf-8")
        Path(tmp, "test_buggy.py").write_text(test_code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_buggy.py"],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "timeout": True, "output": ""}
    return {"passed": proc.returncode == 0, "timeout": False, "output": proc.stdout[-1500:]}


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
