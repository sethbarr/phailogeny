# phailogeny: scoping spec

Status: v0 scope, 2026-09-28. Owner: Seth. Audience: Claude Code, building the MVP.

## 1. Purpose

phailogeny measures how similar AI agents are to one another and turns that into decisions about an agent estate: which agents to merge, which capabilities to build once as a shared service, which forks have drifted, and whether the estate's diversity is growing faster than its usefulness.

Enterprise agent registries (Microsoft Agent 365, AWS Agent Registry, ServiceNow AI Control Tower) record what agents exist, who owns them and what they can access. They do not analyse functional overlap. Consolidation is currently done by manual portfolio review. phailogeny is the analytical layer that sits on top of an inventory.

The MVP is a Python library and CLI that runs on public agent corpora. It is a research tool first. No UI beyond a static HTML report.

## 2. Concepts the code must respect

These come from comparative biology and determine the design.

**Phenetic similarity.** Agents are compared on many characters at once, combined into one distance. This is the core computation.

**Homology vs analogy.** Two similar agents may share ancestry (one was copied or forked from the other, or both from a template) or may have converged independently on the same job. The actions differ: convergent duplicates are merge candidates; related agents that have diverged need their template standardised. The code keeps these two signals separate:
- *Lineage signal*: literal text reuse in prompts and definitions (shingle/MinHash Jaccard).
- *Functional signal*: semantic and capability similarity (embeddings, tool-set overlap).
High functional similarity with low literal overlap indicates convergence. High literal overlap indicates shared ancestry regardless of current function.

**Reticulation.** Agents exchange parts (prompts, tools, retrieval pipelines) across lineages. A strictly bifurcating tree misrepresents this. The MVP produces a neighbour-joining tree for readability plus a cluster assignment; a split network is a later milestone.

**Horizontal transfer.** Relatedness between agents often comes from module-level copying: a prompt section, a tool configuration or a skill moved from one agent into an otherwise unrelated one. Whole-agent forking (vertical descent) also happens, but is probably the less common route. The code therefore treats an agent as a set of modules (the analogue of genes) and measures lineage per module as well as per agent. A module shared at high text identity between agents that are otherwise distant is a candidate transfer, the same logic as gene-tree / species-tree incongruence in bacterial genomics. Two cases need separating:
- *Referenced reuse*: agents point to the same tool, MCP server or skill. This is intended sharing and is healthy.
- *Copied reuse*: agents contain pasted copies of the same text. Fixes do not propagate between copies, so this is maintenance debt and the report flags it.

Two limits on the analogy. Where provenance is recorded (git history, package imports, shared server IDs), use it directly instead of inferring transfer. And shared text frequently comes from a common external source, such as a framework documentation example copied by many people; matching modules against a reference pool of known public templates keeps these from being counted as transfers between agents.

**Pan-genome.** Across an estate, modules fall into *core* (in nearly all agents, usually boilerplate or safety text), *shell* (in many agents across several clusters) and *cloud* (in one or two agents). Shell modules are the main candidates for shared services.

**Genotype vs phenotype.** Declared configuration (prompt, tools, schemas) is the genotype. What an agent does when run is the phenotype. The MVP works on genotype. Milestone 4 measures phenotype with a *common garden* experiment: run every agent on the same fixed battery of probe tasks and record its behaviour, so differences reflect the agent and not its inputs.

**Functional redundancy.** Some duplication is intentional (data boundaries, authority levels). Merge candidates are flagged with any constraint conflicts, never recommended unconditionally.

**Diversity.** Estate diversity is reported as Hill numbers (q = 0, 1, 2) over functional clusters, so sprawl can be tracked over time.

## 3. Test corpora

Priority order. All are public; licences checked where noted.

| # | Corpus | What it gives | Size | Notes |
|---|---|---|---|---|
| 1 | Claude Code subagent collections on GitHub: `wshobson/agents`, `VoltAgent/awesome-claude-code-subagents`, `lst97/claude-code-sub-agents`, `rahulvrane/awesome-claude-agents`, `chusri/claude-code-agents`, plus repos under the GitHub topic `subagents` | Markdown files with YAML frontmatter (`name`, `description`, `tools`, `model`) and a system prompt body | ~200 in wshobson alone (MIT); several hundred to low thousands across repos | **Primary corpus.** Many repos copy from each other (known lineage) and many independently write the same role, e.g. `code-reviewer` (known convergence). This is a natural homology-vs-analogy test. Record repo, path and commit SHA for every agent. |
| 2 | GPTZoo (`security-pride/GPTZoo`) | 730k custom GPTs, 21 attributes, JSON | Large | MIT. Public metadata includes names, descriptions and tool flags. Full instructions and knowledge files need an access request via the project's Google Form. Use metadata only unless Seth obtains access. Good for scale testing and diversity metrics. |
| 3 | A2A agent cards via a2aregistry.org API | Live agent cards: skills, input/output modes, provider | ~390 agents | Closest format to what an enterprise registry holds. Small. Read the API docs before writing the client. |
| 4 | n8n workflow collections on GitHub (e.g. `enescingoz/awesome-n8n-templates`, 280+) | Workflow JSON where nodes are tools and integrations | Hundreds | Tool sets are explicit and structured, so it is a clean test of the capability block. Do **not** scrape n8n.io or its API (robots.txt disallows it); use GitHub-hosted exports only. |
| 5 | Framework example repos: `crewAIInc/crewAI-examples`, LangGraph templates, OpenAI Agents SDK examples | Agents defined in code | Tens | Tests the code-parsing ingesters. Lower priority. |
| 6 | Agent trajectory datasets on Hugging Face (`nebius/SWE-agent-trajectories`, `nebius/SWE-rebench-openhands-trajectories`) | Tool-call traces | Large | One scaffold across many tasks, so the variation is between tasks, not agents. Use only to develop and test trace-feature extraction before Milestone 4. |

Do not use repositories of leaked system prompts.

## 4. Data model

One normalised record per agent. Use `dataclasses` (frozen where sensible). Built-in generics for type hints (`list[str]`, `dict[str, float]`, `X | None`); avoid importing `typing`.

```python
@dataclass(frozen=True)
class AgentProfile:
    agent_id: str            # stable hash of source + path
    name: str
    source: str              # corpus id, e.g. "gh:wshobson/agents"
    source_path: str
    source_commit: str | None
    purpose: str             # description / stated purpose
    prompt: str              # system prompt or instructions body
    tools: frozenset[str]    # normalised tool / integration names
    data_sources: frozenset[str]
    permissions: frozenset[str]
    input_modes: frozenset[str]
    output_modes: frozenset[str]
    model: str | None
    framework: str           # "claude-code", "gpt", "a2a", "n8n", ...
    metadata: dict[str, str] # owner, tags, anything not scored
```

Each agent is also decomposed into modules:

```python
@dataclass(frozen=True)
class AgentModule:
    module_id: str           # hash of normalised content
    agent_id: str
    kind: str                # "prompt_section", "tool", "skill", "subagent_call", "data_source"
    label: str               # heading text, tool name, skill name
    content: str             # normalised text; for tools, the canonical name
    reference: str | None    # set when the module is a pointer to a shared resource
                             # (MCP server id, skill package, tool name) rather than pasted text
```

Missing fields are empty sets or empty strings, never guessed. Tool-name normalisation (e.g. `Read`, `read_file`, `fs.read` to a canonical `file.read`) is a separate, explicit mapping table in `data/tool_aliases.yaml` that can be edited by hand.

## 5. Pipeline and modules

Each module has one job. No module imports from a module later in the pipeline.

```
phailogeny/
  ingest/            # one parser per corpus → list[AgentProfile]
    claude_subagents.py
    gptzoo.py
    a2a_registry.py
    n8n.py
  corpus.py          # load/save profiles as JSONL; dedupe exact copies
  characters/
    boilerplate.py   # strip corpus-common lines/phrases
    modules.py       # split each agent into AgentModule records
    text.py          # embeddings for purpose and prompt
    shingles.py      # MinHash signatures, per agent and per module
    sets.py          # tool / data / permission set features, IDF weights
    templates.py     # match modules against the external reference pool
  distance/
    blocks.py        # one distance matrix per character block
    combine.py       # weighted combination (Gower-style)
  structure/
    cluster.py       # HDBSCAN on combined distance
    tree.py          # neighbour-joining tree
    lineage.py       # homology vs analogy classification of pairs
    families.py      # group modules into families across agents
    transfer.py      # detect candidate horizontal transfers
  analysis/
    merge.py         # merge candidates + constraint conflicts
    pangenome.py     # agent × module-family table; core / shell / cloud
    shared_needs.py  # shell modules and tools spread across distant clusters
    copy_debt.py     # copied (not referenced) modules in several agents
    drift.py         # related agents that diverged
    diversity.py     # Hill numbers, redundancy index
  report/
    html.py          # static HTML report
    newick.py        # tree export
  synth/
    generator.py     # synthetic estates with planted structure
  cli.py
tests/
data/
  tool_aliases.yaml
  reference_templates/   # public example prompts/agents from framework docs
```

### 5.1 Character blocks and distances

| Block | Input fields | Method | Distance |
|---|---|---|---|
| Purpose | `purpose` | Sentence embedding (local `sentence-transformers` model; no paid API in the MVP) | 1 − cosine |
| Prompt | `prompt` after boilerplate stripping | Sentence embedding over chunks, mean-pooled | 1 − cosine |
| Capability | `tools`, `data_sources` | Sets with IDF weighting so near-universal tools (e.g. file read) count less | Weighted Jaccard distance |
| Interface | `input_modes`, `output_modes` | Sets | Jaccard distance |
| Lineage | `prompt` + `purpose` raw text | MinHash over word 5-shingles (`datasketch`) | 1 − estimated Jaccard; **kept out of the combined functional distance** |

Blocks with no data for a pair are dropped from that pair's combined distance and weights renormalised (Gower's rule). Default weights live in a config file and are shown in every report.

**Boilerplate stripping.** Compute document frequency of normalised lines across the corpus; drop lines above a threshold (default: present in >10% of agents in the same source). Implement with string operations and `collections.Counter`; avoid regular expressions.

### 5.2 Structure

- Clustering: `sklearn.cluster.HDBSCAN` with `metric="precomputed"`. Report noise points as singletons.
- Tree: `skbio.tree.nj` on the combined distance. Export Newick.
- Lineage classification for each pair above a functional-similarity threshold:
  - lineage similarity high → `homologous`
  - lineage similarity low, functional similarity high → `convergent`
  Thresholds are calibrated on the synthetic set (Section 6), not hand-picked.

### 5.3 Module-level lineage and transfer

**Splitting.** `characters/modules.py` splits each agent into modules: prompt sections (split at markdown heading lines and blank-line-separated blocks, using `str` methods), each tool, each skill, each call to another agent, each data source. Tools, skills and servers carry a `reference`; prompt sections do not. Section granularity is a parameter (`heading`, `paragraph`) because the right unit is an empirical question (Section 6.4).

**Families.** Modules with MinHash Jaccard ≥ a threshold (default 0.8, calibrated on synthetic data) are grouped into a family by single-linkage over an LSH index (`datasketch.MinHashLSH`). Referenced modules are grouped by canonical reference id.

**External sources.** Modules matching the reference pool in `data/reference_templates/` are tagged `external` and excluded from transfer detection.

**Transfer detection.** For each pair of agents sharing a non-core, non-external family, compare module identity with whole-agent distance. Flag a candidate transfer when the module is near-identical and the agents' vertical lineage distance (computed with shared families masked out, see below) and functional distance are both above threshold. Where git history is available, record the earlier commit as donor; otherwise record the pair as undirected.

**Masked lineage.** Lineage distance between two agents is recomputed after removing modules in families flagged as transferred, analogous to building a species tree from vertically inherited genes only. The model with this masking is **M1**; the whole-agent MinHash lineage of Section 5.1 is **M0**. Both are always computed so they can be compared.

### 5.4 Analyses

- **Merge candidates**: convergent pairs/clusters above threshold. Each one lists shared and differing tools, data sources and permissions. Any difference in permissions or data sources is listed as a constraint conflict.
- **Pan-genome**: agent × family presence table. Families are *core* if in ≥ 90% of agents, *shell* if in ≥ 3 agents across ≥ 2 clusters, *cloud* otherwise (thresholds configurable).
- **Shared needs**: shell families and tools, ranked by the number of clusters they appear in and the mean distance between those clusters. Referenced shell families are existing shared services; copied shell families are candidates to become one.
- **Copy debt**: copied (unreferenced) prompt-section families present in ≥ 3 agents, with the text differences between copies, since drift between copies shows fixes that did not propagate.
- **Drift**: homologous pairs whose functional distance exceeds a threshold.
- **Diversity**: Hill numbers of order 0, 1, 2 over cluster membership; functional redundancy = agents per cluster.

## 6. Validation

The method has to be shown to recover known structure before anyone reads its output on real data.

**6.1 Synthetic estate** (`synth/generator.py`). Generate 150–300 agents from ~20 base roles with:
- *Independent duplicates*: same role written from scratch with different wording (use an LLM to paraphrase at generation time, or hand-write 3–4 variants per role). Ground truth: convergent.
- *Forks*: copy an agent and apply controlled edits (add or remove tools, rewrite n% of the prompt). Ground truth: homologous, with known drift amount.
- *Tool-sharing decoys*: agents with different roles that share an uncommon tool. Ground truth: not merge candidates, but they should show up under shared needs.
- *Horizontal transfers*: copy one prompt section or tool from a donor agent into a recipient in a different role, optionally with light edits. Transfer rate is a generator parameter (fraction of agents receiving ≥ 1 transferred module).
- *External-source copies*: insert the same section from `data/reference_templates/` into several unrelated agents. Ground truth: shared external origin, not transfer.

The generator writes the true vertical genealogy (Newick), the list of planted transfers (donor, recipient, module) and the external-copy list to `synth_truth.json`.

Metrics: adjusted Rand index of clusters against true roles; precision and recall of merge candidates; accuracy of the homologous/convergent labels; rank correlation between planted drift and measured functional distance.

**6.2 Natural experiment** (corpus 1). Build ground truth from the collections:
- Lineage pairs: files with the same name in different repos and near-identical text, or git history showing a copy.
- Convergent pairs: same role name (e.g. `code-reviewer`, `security-auditor`) with low text overlap across repos.
Hand-label a sample of 50 pairs to check these heuristics before trusting them.

**6.3 Ablations.** Rerun with each block removed to show which blocks carry signal. Expect purpose and prompt to dominate on corpus 1, and capability to matter more on n8n.

**6.4 Hypothesis test: does modelling horizontal transfer improve lineage inference?**

*H1.* Masked module-level lineage (M1) recovers true vertical relatedness better than whole-agent lineage (M0), and the advantage grows with transfer rate.

*H0.* M1 performs no better than M0 at any transfer rate, in which case module-level lineage adds complexity without benefit and is dropped from the default pipeline (transfer detection and copy debt may still be kept as standalone reports).

*Design.* Dose-response on synthetic estates. Transfer rate ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}; 20 seeds per level; 200 agents, 20 roles. Also cross with section granularity (`heading`, `paragraph`) and with/without the external reference pool.

*Primary outcome.* Normalised Robinson–Foulds distance between the NJ tree built from each model's lineage distance and the true vertical genealogy (`skbio` `TreeNode.compare_rfd`).

*Secondary outcomes.*
- False homology rate: fraction of pairs related only by transfer that the model labels homologous.
- Transfer detection precision and recall (M1 only).
- External-source false positives: planted external copies flagged as transfers, with and without the reference pool.
- Downstream effect: precision of merge candidates and shared-needs recall under M0 vs M1.

*Analysis.* Paired comparison per seed (M1 − M0) at each transfer rate; report mean difference with bootstrap 95% CI, and plot RF distance against transfer rate for both models. Expected result under H1: similar performance at rate 0 and a widening gap as the rate rises. If M1 is worse at rate 0, that measures the cost of the extra machinery when transfer is absent.

*Decision rule (fixed before running).* Keep M1 as default if its mean normalised RF distance is lower than M0's by ≥ 0.05 with a CI excluding zero at transfer rates ≥ 0.1, and it is no worse than M0 by more than 0.02 at rate 0.

*Natural-corpus check.* On corpus 1, take agents whose git history records a copy of a whole file (vertical) or of a section into a different file (transfer). Report whether M1 separates these better than M0. Treat this as supporting evidence only, since recorded provenance is partial.

*Caveat.* Synthetic transfers follow rules we wrote, so the synthetic result shows the method works when our assumptions hold. The natural-corpus check is the test of whether those assumptions match how people actually copy agents.

## 7. Milestones

1. **Ingest + corpus**: Claude Code subagent parser across ≥4 repos, JSONL corpus, exact-duplicate removal. Tests on fixture files.
2. **Characters + distance + structure**: all five blocks, combined distance, HDBSCAN, NJ tree, Newick export. Module splitting, families, reference-pool matching, transfer detection, and both M0 and M1 lineage.
3. **Synthetic validation + analyses + report**: generator (including transfers and external copies), metrics from 6.1, the H1 test in 6.4, merge / pan-genome / shared-needs / copy-debt / drift / diversity, HTML report. **Checkpoint with Seth before continuing**, including the M0 vs M1 decision.
4. **Common garden phenotype**: fixed battery of ~30 probe tasks; run each Claude Code subagent definition against it in a sandbox; record tool-call sequences and output types; add a behaviour block (tool-call n-gram profile, distribution divergence). Compare genotype-only and genotype+phenotype clustering.
5. **More corpora + network**: GPTZoo metadata, A2A registry, n8n; split network (NeighborNet) if a maintained Python implementation exists, else export distances for SplitsTree.

## 8. CLI

```
phailogeny ingest --source claude-subagents --repo wshobson/agents --out corpus.jsonl
phailogeny build  --corpus corpus.jsonl --config weights.yaml --out run/
phailogeny report --run run/ --out report.html
phailogeny synth  --roles 20 --agents 200 --seed 7 --out synth.jsonl
phailogeny validate --corpus synth.jsonl --truth synth_truth.json
phailogeny experiment h1 --rates 0,0.05,0.1,0.2,0.3,0.5 --seeds 20 --out h1/
```

## 9. Code conventions

- Python 3.12. `uv` for environment and dependencies; `pyproject.toml`.
- Type hints on every function and method, using built-in generics; avoid importing `typing`.
- Google-style docstrings on every function, method and class.
- No `abc` module; ingesters are plain classes that share a documented method signature.
- No nested functions.
- Avoid regular expressions; use `str` methods, `yaml`, `json`, `tomllib`.
- No deprecated APIs.
- Separation of concerns as in Section 5; each function does one thing.
- `pytest`; fixtures in `tests/fixtures/`. `ruff` for lint and format.
- Deterministic: every stochastic step takes a seed.

Dependencies (MVP): `numpy`, `scipy`, `scikit-learn`, `scikit-bio`, `sentence-transformers`, `datasketch`, `pyyaml`, `jinja2`, `httpx`, `pytest`, `ruff`.

## 10. Non-goals for the MVP

Enterprise connectors (Agent 365, AWS, ServiceNow), live telemetry ingestion, a web app, automated merging of agents, and paid embedding APIs.

## 11. Open questions for Seth

1. Default block weights: equal, or purpose-heavy?
2. Request full GPTZoo instruction access, or stay with metadata?
3. For Milestone 4, which model and sandbox to run the common-garden probes on, and what budget?
4. Open-source licence for phailogeny itself.
5. Which public templates go into the external reference pool (Anthropic, OpenAI, CrewAI, LangGraph documentation examples are the obvious start).
6. Is the 0.05 RF improvement threshold in 6.4 the right bar, or should the decision rest on the downstream merge-candidate precision instead?
