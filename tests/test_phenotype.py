"""Tests for the common-garden pipeline, using a fake model call (no network)."""

from __future__ import annotations

import json

import numpy as np

from phailogeny.phenotype import common_garden as cg


def _records(n: int = 6) -> list[dict]:
    roles = ["reviewer", "debugger", "writer"]
    return [
        {
            "agent_id": f"r{i % 2}/{roles[i // 2]}",
            "source": f"r{i % 2}",
            "source_path": f"r{i % 2}/{roles[i // 2]}.md",
            "purpose": roles[i // 2],
            "prompt": f"You are a {roles[i // 2]} from repo {i % 2}.",
        }
        for i in range(n)
    ]


def _fake_call(system, prompt, model, effort, schema=None):
    if schema:
        verdict = {f: False for f in cg.JUDGE_FIELDS} | {"caught_planted_issue": "n/a", "approach": "x"}
        return {"ok": True, "structured": verdict, "text": "", "model": model, "cost_usd": 0.0, "usage": {}}
    return {"ok": True, "text": f"{system} answers: {prompt[:30]}", "model": model, "cost_usd": 0.001, "usage": {"input": 10, "output": 5}}


def _probes(tmp_path):
    path = tmp_path / "probes.yaml"
    path.write_text(
        "version: 1\nprobes:\n  - {id: p1, group: g, planted: null, prompt: 'Review this.'}\n"
        "  - {id: p2, group: g, planted: 'bug', prompt: 'Debug this.'}\n",
        encoding="utf-8",
    )
    return path


def test_plan_run_resume_and_analyse(tmp_path) -> None:
    records = _records()
    n = len(records)
    functional = np.full((n, n), 0.5)
    np.fill_diagonal(functional, 0.0)
    runs = tmp_path / "runs"
    run_dir = cg.plan_run(records, functional, np.zeros((n, n)), _probes(tmp_path), n_agents=6, replicates=2, runs_dir=runs)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["n_requests"] == 7 * 2 * 2  # 6 agents + baseline, 2 probes, 2 reps
    assert manifest["strata"]["twin"] >= 2

    first = cg.run_cli(run_dir, workers=3, limit=5, runs_dir=runs, call=_fake_call)
    second = cg.run_cli(run_dir, workers=3, runs_dir=runs, call=_fake_call)
    assert first["done"] == 5 and second["done"] == manifest["n_requests"] - 5  # resumes, no repeats
    judged = cg.run_cli(run_dir, phase="judge", workers=3, runs_dir=runs, call=_fake_call)
    assert judged["done"] == manifest["n_requests"]

    analysis = cg.analyse_run(run_dir, runs)
    assert analysis["coverage"] == 1.0 and analysis["judged"] == 1.0
    assert (run_dir / "report.md").exists()
    assert len((runs / "index.jsonl").read_text().splitlines()) == 1


def test_editing_a_probe_changes_the_key() -> None:
    probe = {"hash": "a"}
    edited = {"hash": "b"}
    settings = {"backend": "claude-cli", "effort": "low", "max_tokens": None}
    assert cg.response_key("m", settings, "sys", probe, 0) != cg.response_key("m", settings, "sys", edited, 0)


def test_mantel_detects_identical_matrices() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(12, 3))
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    assert cg.mantel(d, d, permutations=99)["r"] == 1.0
