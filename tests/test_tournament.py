"""Tests for clade-tournament preparation, with a fake model (no network)."""

from __future__ import annotations

import json

from skbio import TreeNode

from phailogeny.phenotype.tournament import prepare, validate_task

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from buggy import *\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _verifiable(**overrides) -> dict:
    task = {"id": "t", "type": "verifiable", "title": "", "prompt": f"Fix:\n```python\n{BUGGY}```",
            "buggy_code": BUGGY, "test_code": TEST, "reference_code": FIXED, "rubric": [],
            "expected_behaviour": "", "variant_of": ""}
    return task | overrides


def test_valid_verifiable_task_fails_on_bug_and_passes_on_fix() -> None:
    assert validate_task(_verifiable())["valid"]


def test_task_whose_test_misses_the_bug_is_rejected() -> None:
    result = validate_task(_verifiable(test_code="from buggy import *\n\ndef test_ok():\n    assert True\n"))
    assert not result["valid"] and "bug not caught" in result["reasons"][0]


def test_prepare_writes_reviewable_files(tmp_path) -> None:
    tree = TreeNode.read(["((a/x:1,b/x:1):1,(a/y:1,c/z:1):1);"])
    records = [{"agent_id": n, "source": n.split("/")[0], "source_path": f"{n}.md", "purpose": "debugs", "prompt": "fix bugs"}
               for n in ["a/x", "b/x", "a/y", "c/z"]]

    def fake_call(system, prompt, model, effort, schema=None):
        if "capabilities" in schema["properties"]:
            out = {"capabilities": [{"id": "fix", "name": "Fix bugs", "definition": "Finds and fixes bugs"}],
                   "claims": [{"agent_id": r["agent_id"], "capability_ids": ["fix"]} for r in records]}
        else:
            out = {"tasks": [_verifiable(id="fix_1")]}
        return {"ok": True, "structured": out, "cost_usd": 0.01}

    report = prepare(tree, "a/x", records, tmp_path / "out", tmp_path / "log.jsonl", tasks_per_capability=1,
                     workers=1, holdout_ids=["A2_debug"], call=fake_call)
    assert report["members"] == 4 and report["tasks_by_type"]["verifiable"] == {"total": 1, "valid": 1}
    assert report["tasks_by_type"]["holdout"]["total"] == 1
    assert json.loads((tmp_path / "out" / "claims.json").read_text())["capabilities"][0]["id"] == "fix"


def test_run_and_rank_with_a_fake_model(tmp_path) -> None:
    import yaml

    from phailogeny.phenotype.tournament import rank, run_verifiable

    data, runs = tmp_path / "data", tmp_path / "runs"
    (data / "t").mkdir(parents=True)
    (data / "t" / "members.json").write_text(json.dumps({"members": [{"agent_id": "a/good"}, {"agent_id": "a/bad"}]}))
    task = _verifiable(id="fix_1", capability="fix", validation={"valid": True, "reasons": []})
    (data / "t" / "tasks.yaml").write_text(yaml.safe_dump({"tasks": [task]}))
    records = [{"agent_id": "a/good", "prompt": "good"}, {"agent_id": "a/bad", "prompt": "bad"}]

    def fake_call(system, prompt, model, effort, schema=None):
        code = FIXED if system in ("good", "You are a helpful assistant.") else BUGGY
        return {"ok": True, "text": f"Here:\n```python\n{code}```", "cost_usd": 0.01}

    status = run_verifiable("t", records, replicates=2, workers=2, call=fake_call, data_root=data, runs_root=runs)
    assert status["done"] == 8  # 2 members + 2 controls, 1 task, 2 reps
    again = run_verifiable("t", records, replicates=2, workers=2, call=fake_call, data_root=data, runs_root=runs)
    assert again["queued"] == 0  # resumes; nothing paid twice
    result = rank("t", model="claude-opus-5-5", runs_root=runs, data_root=data, n_boot=50)
    scores = {row["agent_id"]: row["pass_rate"] for row in result["ranking"]}
    assert scores["a/good"] == 1.0 and scores["a/bad"] == 0.0 and scores["control/no-prompt"] == 1.0


def test_extract_code_prefers_the_module_over_a_trailing_test_block() -> None:
    from phailogeny.phenotype.tournament import extract_code

    answer = f"Fix:\n```python\n{FIXED}```\nAnd a regression test:\n```python\n{TEST}```"
    assert extract_code(answer, FIXED).strip() == FIXED.strip()


def test_extract_code_handles_four_backtick_fences() -> None:
    from phailogeny.phenotype.tournament import extract_code

    answer = f"Snippet:\n```python\nx = 1\n```\nFull module:\n````python\n{FIXED}````\n"
    assert extract_code(answer, FIXED).strip() == FIXED.strip()


def test_multi_file_answer_overlays_changed_files_and_runs_tests() -> None:
    from phailogeny.phenotype.tournament import score_answer

    task = {"type": "verifiable_multi",
            "files": [{"path": "calc.py", "content": "from helpers import sign\n\ndef add(a, b):\n    return a + sign(b) * b\n"},
                      {"path": "helpers.py", "content": "def sign(x):\n    return -1\n"}],
            "test_code": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"}
    fixed = "```python\n# file: helpers.py\ndef sign(x):\n    return 1\n```"
    assert score_answer(task, fixed)["passed"]
    assert not score_answer(task, "No change needed.")["passed"]


def test_unlabelled_snippet_does_not_overwrite_a_file_when_answer_labels_files() -> None:
    from phailogeny.phenotype.tournament import extract_files

    original = {"a.py": "def f():\n    return 1\n\ndef g():\n    return 2\n", "b.py": "def h():\n    return 3\n"}
    answer = "Look at this:\n```python\ndef h():\n    pass\n```\nFix:\n```python\n# file: a.py\ndef f():\n    return 10\n\ndef g():\n    return 2\n```"
    assert set(extract_files(answer, original)) == {"a.py"}


def test_wilson_interval_is_honest_at_a_perfect_score() -> None:
    from phailogeny.phenotype.tournament import wilson_interval

    low, high = wilson_interval(26, 26)
    assert 0.86 < low < 0.88 and high == 1.0
