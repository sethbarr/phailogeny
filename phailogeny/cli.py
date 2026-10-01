"""Command-line interface for phailogeny."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phailogeny.analysis.h1_experiment import h1_summary
from phailogeny.analysis.real_agents import load_real_agent_records, summarize_real_agents
from phailogeny.corpus import load_jsonl, write_jsonl
from phailogeny.ingest.claude_subagents import ClaudeSubagentIngester
from phailogeny.report.html import render_html_report
from phailogeny.synth.generator import SyntheticEstateGenerator


def _ingest_agents(source_dir: str, out_path: str) -> dict[str, object]:
    """Parse a directory of agent markdown files and write a JSONL corpus."""
    records = load_real_agent_records(source_dir, source="claude-subagents")
    write_jsonl(out_path, records)
    return {"ingested": len(records), "out": out_path}


def _build_summary(corpus_path: str) -> dict[str, object]:
    """Load a JSONL corpus and calculate a simple summary."""
    records = load_jsonl(corpus_path)
    summary = summarize_real_agents(records)
    return summary


def _report_summary(summary: dict[str, object], out_path: str) -> str:
    """Render a static HTML summary report and write it to disk."""
    content = json.dumps(summary, sort_keys=True, indent=2)
    html = render_html_report("phailogeny report", {"summary": content})
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


def _synth_agents(roles: int, agents_per_role: int, seed: int, out_path: str) -> list[dict[str, object]]:
    """Generate a synthetic estate and write it to JSONL."""
    generator = SyntheticEstateGenerator(roles=roles, agents_per_role=agents_per_role, seed=seed)
    agents = generator.generate()
    records = [
        {
            "agent_id": f"synthetic-{index}",
            "name": f"{agent.role}-{index}",
            "source": "synthetic",
            "source_path": f"synthetic/{agent.role}/{index}.md",
            "source_commit": None,
            "purpose": agent.role,
            "prompt": agent.prompt,
            "tools": list(agent.tools),
            "data_sources": [],
            "permissions": [],
            "input_modes": [],
            "output_modes": [],
            "model": None,
            "framework": "synthetic",
            "metadata": {"role": agent.role},
        }
        for index, agent in enumerate(agents)
    ]
    write_jsonl(out_path, records)
    return records


def _validate_synth(corpus_path: str, truth_path: str | None = None) -> dict[str, float]:
    """Validate a synthetic corpus against a truth JSON file when available."""
    records = load_jsonl(corpus_path)
    if truth_path is None:
        return {"accuracy": 1.0 if records else 0.0}

    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    labels = [str(record.get("purpose") or "") for record in records]
    truth_labels = [str(item.get("role") or "") for item in truth.get("agents", [])]
    accuracy = sum(1 for left, right in zip(labels, truth_labels) if left == right) / len(labels) if labels else 0.0
    return {"accuracy": accuracy}


def _run_experiment(rates: str, seeds: int) -> dict[str, object]:
    """Run the placeholder H1 experiment summary for synthetic transfer rates."""
    items: list[dict[str, float]] = []
    for rate in [float(value.strip()) for value in rates.split(",") if value.strip()]:
        for _ in range(seeds):
            items.append({"m0": 0.8 - rate * 0.2, "m1": 0.75 - rate * 0.25})
    return h1_summary(items)


def _write_tree_outputs(result: dict[str, object], out: Path, top: int) -> dict[str, object]:
    """Write tree.nwk, distances.csv, pairs.json and report.html for one analysis."""
    import numpy as np

    from phailogeny.estate import nearest_neighbour_of_each, nearest_pairs
    from phailogeny.report.tree_report import render_tree_report

    out.mkdir(parents=True, exist_ok=True)
    result["tree"].write(str(out / "tree.nwk"))
    ids = result["ids"]
    np.savetxt(out / "distances.csv", result["functional"], delimiter=",", fmt="%.4f", header=",".join(ids), comments="")
    pairs = nearest_pairs(result, limit=top)
    groups: dict[int, list[str]] = {}
    for agent, label in zip(ids, result["cluster_labels"]):
        if label >= 0:
            groups.setdefault(label, []).append(agent)
    clusters = list(groups.values())
    (out / "pairs.json").write_text(
        json.dumps({"pairs": pairs, "nearest": nearest_neighbour_of_each(result), "clusters": clusters}, indent=2),
        encoding="utf-8",
    )
    (out / "report.html").write_text(render_tree_report(result, pairs, clusters), encoding="utf-8")
    return {"clusters": len(clusters), "out": str(out)}


def _write_figures(centred: dict[str, object], uncentred: dict[str, object], out: Path) -> None:
    """Circular overview SVGs of both trees, plus a labelled zoom on the debugging clade."""
    from phailogeny.report.figures import clade_around, radial_svg, rect_svg

    out.mkdir(parents=True, exist_ok=True)
    source_of = {aid: str(rec["source"]) for aid, rec in zip(centred["ids"], centred["records"])}
    n = len(source_of)
    for tag, result in (("centred", centred), ("uncentred", uncentred)):
        label = "style-centred" if tag == "centred" else "uncentred"
        title = f"{n:,} agents, NJ on {label} embedding distances"
        (out / f"tree_{tag}.svg").write_text(radial_svg(result["tree"], source_of, title=title), encoding="utf-8")
    names = {str(tip.name) for tip in centred["tree"].tips()}
    anchor = next((a for a in ("wshobson/debugging-toolkit-debugger", "voltagent/debugger") if a in names), None)
    if anchor:
        clade = clade_around(centred["tree"], anchor, 18, 40).copy()
        title = "Zoom: the debugging clade (style-centred tree), coloured by repo"
        (out / "clade_debugging.svg").write_text(rect_svg(clade, source_of, title=title), encoding="utf-8")


def _tree(dirs: list[str], corpus: str | None, out_dir: str, top: int, weights: str) -> dict[str, object]:
    """Build the style-centred tree (primary) and the uncentred tree, and compare them.

    out/          style-centred analysis (tree.nwk, distances.csv, pairs.json, report.html)
    out/uncentred/ the same outputs without style centring, kept as a method baseline
    out/tree_comparison.json  topology, cophenetic, cluster and nearest-neighbour agreement
    """
    from phailogeny.estate import analyse_estate, dedupe_copies, load_subagent_dirs
    from phailogeny.structure.compare import compare_analyses

    records = load_jsonl(corpus) if corpus else load_subagent_dirs(dirs)
    loaded = len(records)
    records = dedupe_copies(records)
    parsed = {key: float(val) for key, val in (item.split("=") for item in weights.split(",") if item)}
    centred = analyse_estate(records, weights=parsed, style_centring=True)
    uncentred = analyse_estate(records, weights=parsed, style_centring=False)

    out = Path(out_dir)
    primary = _write_tree_outputs(centred, out, top)
    _write_tree_outputs(uncentred, out / "uncentred", top)
    _write_figures(centred, uncentred, out / "figures")
    comparison = compare_analyses(uncentred, centred, "uncentred", "centred")
    comparison["style_centring"] = centred["style_centring"]
    (out / "tree_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return {
        "loaded": loaded,
        "after_dedupe": len(records),
        "clusters": primary["clusters"],
        "coverage": centred["coverage"],
        "comparison": {k: v for k, v in comparison.items() if k != "nearest_neighbour_changed_examples"},
        "out": str(out),
    }


def _garden(args: argparse.Namespace) -> dict[str, object]:
    """Plan, run, judge and analyse a common-garden phenotype experiment."""
    import anthropic

    from phailogeny.estate import analyse_estate, dedupe_copies, load_subagent_dirs
    from phailogeny.phenotype import common_garden as cg

    runs_dir = Path(args.runs)
    if args.garden_command == "plan":
        records = dedupe_copies(load_subagent_dirs(args.dir))
        result = analyse_estate(records, weights={"purpose": 1, "prompt": 1, "capability": 0, "interface": 1})
        commits = {}
        for directory in args.dir:
            head = Path(directory) / ".git"
            if head.exists():
                import subprocess

                commits[Path(directory).name] = subprocess.run(
                    ["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip()
        run_dir = cg.plan_run(
            records, result["functional"], result["lineage"], args.probes,
            n_agents=args.agents, replicates=args.replicates, model=args.model, effort=args.effort,
            backend=args.backend, judge_model=args.judge_model or args.model, seed=args.seed,
            runs_dir=runs_dir, commits=commits,
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        return {"run": str(run_dir), "requests": manifest["n_requests"], "strata": manifest["strata"], "estimate": manifest["estimate"]}

    run_dir = Path(args.run)
    if args.garden_command == "run":
        return cg.run_cli(run_dir, phase="respond", workers=args.workers, limit=args.limit, runs_dir=runs_dir)
    if args.garden_command == "judge":
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["settings"]["backend"] == "claude-cli":
            return cg.run_cli(run_dir, phase="judge", workers=args.workers, limit=args.limit, runs_dir=runs_dir)
        return cg.submit_judgments(run_dir, anthropic.Anthropic(), runs_dir)
    if args.garden_command == "submit":
        return cg.submit_responses(run_dir, anthropic.Anthropic(), runs_dir)
    if args.garden_command == "collect":
        return cg.collect(run_dir, anthropic.Anthropic(), runs_dir)
    if args.garden_command == "analyse":
        analysis = cg.analyse_run(run_dir, runs_dir)
        return {"report": str(run_dir / "report.md"), "coverage": analysis["coverage"], "mantel": analysis["mantel"]}
    raise ValueError(args.garden_command)


def _tournament(args: argparse.Namespace) -> dict[str, object]:
    """Prepare a clade tournament: claim matrix, generated tasks, local validation."""
    from skbio import TreeNode

    from phailogeny.estate import assign_ids, dedupe_copies, load_subagent_dirs
    from phailogeny.phenotype.tournament import prepare, rank, run_verifiable

    if args.tournament_command == "rank":
        result = rank(args.name, model=args.model)
        return {k: v for k, v in result.items() if k != "task_difficulty"}
    records = dedupe_copies(load_subagent_dirs(args.dir))
    assign_ids(records)
    if args.tournament_command == "run":
        effort = None if args.effort == "none" else args.effort
        return run_verifiable(args.name, records, replicates=args.replicates, workers=args.workers, limit=args.limit,
                              model=args.model, effort=effort)
    tree = TreeNode.read(args.tree)
    return prepare(
        tree, args.anchor, records,
        out_dir=Path("data/tournaments") / args.name,
        log_path=Path("runs/tournaments") / args.name / "generation.jsonl",
        tasks_per_capability=args.tasks_per_capability,
        workers=args.workers,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="phailogeny", description="Analyze agent estates.")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest agent definitions from a source corpus.")
    ingest_parser.add_argument("--source", default="claude-subagents")
    ingest_parser.add_argument("--dir", required=True, help="Directory of markdown agent files.")
    ingest_parser.add_argument("--out", required=True, help="Output corpus JSONL path.")

    build_parser_cmd = subparsers.add_parser("build", help="Build a distance model from a corpus.")
    build_parser_cmd.add_argument("--corpus", required=True, help="Input corpus JSONL path.")

    report_parser = subparsers.add_parser("report", help="Render a static HTML report.")
    report_parser.add_argument("--corpus", required=True, help="Input corpus JSONL path.")
    report_parser.add_argument("--out", required=True, help="Output HTML path.")

    synth_parser = subparsers.add_parser("synth", help="Generate synthetic estates for validation.")
    synth_parser.add_argument("--roles", type=int, default=20)
    synth_parser.add_argument("--agents", type=int, default=200)
    synth_parser.add_argument("--seed", type=int, default=7)
    synth_parser.add_argument("--out", required=True, help="Output JSONL path.")

    validate_parser = subparsers.add_parser("validate", help="Validate a synthetic estate against ground truth.")
    validate_parser.add_argument("--corpus", required=True)
    validate_parser.add_argument("--truth", default=None)

    experiment_parser = subparsers.add_parser("experiment", help="Run the transfer hypothesis experiment.")
    experiment_parser.add_argument("--rates", default="0,0.05,0.1,0.2,0.3,0.5")
    experiment_parser.add_argument("--seeds", type=int, default=20)

    tree_parser = subparsers.add_parser("tree", help="Build the agent similarity tree and report.")
    tree_parser.add_argument("--dir", action="append", default=[], help="Subagent repo dir (repeatable).")
    tree_parser.add_argument("--corpus", default=None, help="Or a JSONL corpus instead of --dir.")
    tree_parser.add_argument("--out", default="out", help="Output directory.")
    tree_parser.add_argument("--top", type=int, default=50, help="Closest pairs to list.")
    tree_parser.add_argument(
        "--weights",
        default="purpose=1,prompt=1,capability=0,interface=1",
        help="Block weights. Capability defaults to 0: Claude subagent `tools` are permissions "
        "that follow repo house style, not function.",
    )

    garden = subparsers.add_parser("garden", help="Common-garden phenotype experiment.")
    garden_sub = garden.add_subparsers(dest="garden_command", required=True)
    plan = garden_sub.add_parser("plan", help="Sample agents and freeze a run (no model calls).")
    plan.add_argument("--dir", action="append", default=[], required=True)
    plan.add_argument("--probes", default="data/probes.yaml")
    plan.add_argument("--agents", type=int, default=60)
    plan.add_argument("--replicates", type=int, default=2)
    plan.add_argument("--model", default="claude-opus-5-5")
    plan.add_argument("--judge-model", default=None, help="Defaults to --model.")
    plan.add_argument("--effort", default="low")
    plan.add_argument("--backend", choices=["claude-cli", "anthropic-batch"], default="claude-cli")
    plan.add_argument("--seed", type=int, default=7)
    for name, help_text in [
        ("run", "Run missing responses through local sandboxed Claude Code, in parallel."),
        ("judge", "Grade collected responses."),
        ("submit", "anthropic-batch backend: submit a batch."),
        ("collect", "anthropic-batch backend: fetch finished batches."),
        ("analyse", "Compare behaviour with genotype; write analysis.json and report.md."),
    ]:
        sub = garden_sub.add_parser(name, help=help_text)
        sub.add_argument("--run", required=True, help="Run directory, e.g. runs/2026-09-28-ab12cd34")
        sub.add_argument("--workers", type=int, default=6)
        sub.add_argument("--limit", type=int, default=None, help="Only do this many calls (smoke tests).")
    for sub in garden_sub.choices.values():
        sub.add_argument("--runs", default="runs", help="Root for the shared store and run folders.")

    tournament = subparsers.add_parser("tournament", help="Rank agents within a clade on a shared test battery.")
    tournament_sub = tournament.add_subparsers(dest="tournament_command", required=True)
    prep = tournament_sub.add_parser("prepare", help="Claim matrix, task generation and local validation.")
    prep.add_argument("--dir", action="append", default=[], required=True)
    prep.add_argument("--tree", default="out/tree.nwk")
    prep.add_argument("--anchor", required=True, help="An agent id inside the clade.")
    prep.add_argument("--name", required=True, help="Output folder name under data/tournaments/.")
    prep.add_argument("--tasks-per-capability", type=int, default=6)
    prep.add_argument("--workers", type=int, default=6)
    run = tournament_sub.add_parser("run", help="Members and controls answer verifiable tasks; tests score them (sandboxed).")
    run.add_argument("--dir", action="append", default=[], required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--replicates", type=int, default=2)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--limit", type=int, default=None, help="Only this many calls (smoke test).")
    run.add_argument("--model", default="claude-opus-5-5", help="The common-garden model every agent runs on.")
    run.add_argument("--effort", default="low", help="Effort level, or 'none' to omit it.")
    ranker = tournament_sub.add_parser("rank", help="Pass rates with bootstrap confidence intervals.")
    ranker.add_argument("--name", required=True)
    ranker.add_argument("--model", default="claude-opus-5-5")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the phailogeny CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "ingest":
        result = _ingest_agents(args.dir, args.out)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "build":
        result = _build_summary(args.corpus)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    if args.command == "report":
        summary = _build_summary(args.corpus)
        out_path = _report_summary(summary, args.out)
        print(json.dumps({"report": out_path}, sort_keys=True))
        return 0

    if args.command == "synth":
        records = _synth_agents(args.roles, args.agents // max(args.roles, 1), args.seed, args.out)
        print(json.dumps({"generated": len(records), "out": args.out}, sort_keys=True))
        return 0

    if args.command == "validate":
        result = _validate_synth(args.corpus, args.truth)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "experiment":
        result = _run_experiment(args.rates, args.seeds)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    if args.command == "tree":
        if not args.dir and not args.corpus:
            parser.error("tree needs --dir or --corpus")
        result = _tree(args.dir, args.corpus, args.out, args.top, args.weights)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    if args.command == "tournament":
        print(json.dumps(_tournament(args), sort_keys=True, indent=2))
        return 0

    if args.command == "garden":
        print(json.dumps(_garden(args), sort_keys=True, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
