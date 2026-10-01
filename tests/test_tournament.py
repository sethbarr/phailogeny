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
