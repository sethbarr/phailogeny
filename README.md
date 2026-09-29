# phailogeny

Comparative "phylogenetics" for AI agents: measure how similar agent definitions are, build a tree,
and separate shared ancestry (copying) from convergence (independently written for the same job).
The design borrows from comparative biology; see [SPEC.md](SPEC.md) for the full rationale.

Two motivations, two measures:

- **For organisations running many agents:** which agents do the same job? That is a merge or
  shared-service question, and it doesn't matter whether two agents converged or one was copied.
  Measure: style-centred embedding similarity.
- **Phylogeny done right:** separating descent from convergence. Function similarity cannot do
  this (finding 5); shared rare features can (finding 6).

Status: research prototype, September 2026. Everything below runs locally except the optional
behaviour experiment (`phailogeny garden`), which calls Claude.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .   # Python 3.12+
bash scripts/fetch_corpora.sh                      # clones 10 public agent repos into data/raw/
.venv/bin/phailogeny tree $(for d in data/raw/*/; do printf -- "--dir %s " "$d"; done) --out out
.venv/bin/python -m pytest -q
```

Outputs (already committed in `out/` so you can look without running anything):

| File | What |
| --- | --- |
| `out/tree.nwk` | Neighbour-joining tree, style-centred distances, midpoint-rooted (open in iTOL / FigTree) |
| `out/uncentred/tree.nwk` | Same, without style centring (method baseline) |
| `out/tree_comparison.json` | Robinson-Foulds, cophenetic r, cluster ARI between the two trees |
| `out/report.html` | Tree coloured by source repo, closest pairs, clusters |
| `out/traits/*.json` | Evaluations of each distance measure and each centring method |

## Data

1,178 agent definitions (1,003 after collapsing exact duplicates) from 10 public GitHub repos:
Claude Code subagents (markdown + YAML frontmatter) and one set of OpenAI Codex subagents (TOML).
Each agent = a one-line description + a system prompt, sometimes a tool list.

## Method

**Taxa** are agents. **Characters** are blocks of the definition, each turned into a distance:

| Block | Method | Role |
| --- | --- | --- |
| Description | Sentence embedding (all-MiniLM-L6-v2, local), cosine distance | Declared function |
| Prompt | Same model; lines shared by >10% of a repo removed; 150-word chunks averaged | Declared function |
| Text overlap | MinHash Jaccard on word 5-shingles | Ancestry (copying); **kept out of the tree** |
| Tools | IDF-weighted Jaccard | Weight 0 here: in this corpus it records permissions, not function |

Functional distance = mean of description and prompt distances. Tree = neighbour-joining
(scikit-bio), midpoint-rooted. Clusters = HDBSCAN on the same distances.

**Style centring.** Each repo writes prompts from its own template, and embeddings pick that up:
56% of agents' nearest neighbours are in their own repo (15% expected). We treat same-named
agents in different repos ("twins", e.g. two `debugger.md`) as sharing a job, so the vector gap
between twins is mostly style. A per-repo offset is fitted by ridge least squares over all twin
pairs and subtracted before computing distances (`phailogeny/estate.py: fit_style_offsets`).
Repos with few twins get offsets shrunk toward zero, so single-domain repos keep their domain.

**Validation without ground truth.** For each twin, rank it among all agents of its repo by
distance from its partner; report how often it comes first. 5-fold cross-validation (folds by
name) when the method itself is fitted on twins.

**Behaviour ("common garden").** Planned, not yet run at scale: every sampled agent answers the
same 20 probe tasks (`data/probes.yaml`) on the same model, twice; answers are compared by
embedding, formatting and an LLM judge; behaviour distance is tested against genotype distance
with a Mantel test (`phailogeny/phenotype/common_garden.py`).

## Findings so far

1. **Copying comes in three levels.** Verbatim pastes within repos (175 extra copies); lightly
   edited copies between repos (wshobson -> buildwithclaude, 5-shingle overlap up to 0.28); and
   full rewrites that keep the roster but no text (wshobson -> lst97, documented in lst97's README;
   VoltAgent Claude -> VoltAgent Codex, 154 shared names, max overlap 0.05).
2. **Text overlap does not track function.** Finding a rewritten twin by text overlap is at
   chance (3-7% found first). Embeddings find it first 77-83% of the time.
3. **Keyword trait profiles did worse than embeddings** (32% vs 77%): generic traits are near
   universal, and terse prompts yield few hits, so traits were *more* format-sensitive.
4. **Twin-based style centring helps** (CV): twin in global top 5 rises 50% -> 73%; same-repo
   nearest neighbour 56% -> 40%. The centred and uncentred trees agree on overall shape
   (cophenetic r 0.90) but differ in 57% of splits (RF), and clusters agree at ARI 0.50.

5. **Controlled paraphrase test** (`scripts/paraphrase_test.py`; 50 wshobson agents, each
   rewritten by Claude three ways; the true source is known). Rank of the source among all 1,003
   agents:

   | Rewrite | 5-gram overlap with source | Text overlap finds source first | Embeddings find source first (top 5) |
   | --- | --- | --- | --- |
   | Light edit | 0.49 | 100% | 100% (100%) |
   | Full rewrite | 0.00 | 32% (median rank 502, chance) | 92% (100%) |
   | Codex-style checklist | 0.00 | 4% | 82% (100%) |

   Embeddings recover the source, but they cannot say *whether* a pair is a copy: rewrite-to-source
   similarity (median 0.81) overlaps natural same-name pairs that are presumably independent
   (median 0.73, 10th-90th percentile 0.61-0.84). Function similarity detects "same job", not
   "same ancestry"; once text is rewritten, homology needs other evidence (see question 3).
   Distributions: `out/paraphrase/similarity_distributions.json`.

6. **Shared rare features separate copies from convergence** (`scripts/rare_features.py`,
   offline). Score = IDF-weighted count of terms two agents share that at most 3 other agents
   use (the pair itself excluded from rarity). Known rewrites keep idiosyncratic details of their
   source (`asnotracking`, `benchmarkdotnet`, `webapplicationfactory` for a .NET agent) that
   independent same-job agents don't. At matched functional similarity (+/-0.03), separating
   known copies from presumed-independent twins:

   | Copies | AUC, shared rare features | AUC, functional similarity |
   | --- | --- | --- |
   | Full rewrites | 0.90 | 0.56 |
   | Codex-style rewrites | 0.87 | 0.47 |

   The natural data agree: documented rewrites (lst97 from wshobson) share a median rare score of
   54, against 12 for other same-name pairs and 6 for close non-twin pairs. Caveats: the synthetic
   copies were written by an LLM that saw the source, which may keep more detail than a human
   would; and "other repos" twins may include undocumented copies, which would make the
   comparison conservative. In phylogenetic terms this is synapomorphy counting with rarity as a
   proxy for "derived", so homoplasy in rare domain terms (two .NET agents both citing Dapper) is
   the main risk.

## Questions for a phylogeneticist

1. **Tree or network?** Agents swap parts across lineages (prompt sections, tool configs). Is NJ
   defensible for a readable summary, or should the primary figure be a split network
   (NeighborNet)? How would you test treeness here (delta scores, Q-residuals)?
2. **Embedding distances are not additive.** Cosine distances between embeddings have no
   evolutionary model behind them. Is NJ on non-additive distances acceptable if framed as
   phenetic clustering rather than phylogeny? Better alternatives?
3. **Homology.** "Same filename in another repo" is the homology proxy. Is there a better way to
   assert homology (e.g. shared roster order, shared section skeleton), and how would you
   formalise homology vs convergence given only the endpoints?
4. **Repo style as non-independence.** Style centring resembles removing a shared-ancestry
   effect before comparing traits. Is there a comparative-methods analogue (PGLS-like, random
   effect per repo) that would be more principled than subtracting fitted offsets?
5. **Rare features as synapomorphies.** Is corpus rarity a reasonable proxy for derived
   character states? How would you model homoplasy in rare domain vocabulary?
6. **Support values.** What would a sensible bootstrap be? Resampling prompt chunks, probe tasks
   (for behaviour), or embedding dimensions?
7. **Mantel test** for genotype vs behaviour: fine here, or use something less criticised?
8. **Rooting.** Midpoint for now. Would an outgroup (a clearly unrelated domain such as the
   security-testing repo) be more meaningful?

## Layout

```
phailogeny/
  ingest/        parsers: Claude subagents (md), Codex subagents (toml), GPTZoo, A2A, n8n
  estate.py      loading, dedupe, embeddings, style centring, distances, NJ tree, HDBSCAN
  characters/    trait taxonomy extraction, module splitting
  structure/     tree comparison, lineage classification
  analysis/      twin-retrieval evaluation and older analyses
  phenotype/     common-garden behaviour experiment
  report/        HTML tree report
scripts/         corpus fetch and the evaluation scripts behind every number above
data/            probes.yaml, traits.yaml, tool_aliases.yaml
```

Corpus licences: most repos are MIT. `contains-studio/agents` has no licence; it is analysed in
aggregate and its text is not redistributed here.

## Licence

Code: [Apache-2.0](LICENSE). Copyright 2026 Seth Barribeau.
