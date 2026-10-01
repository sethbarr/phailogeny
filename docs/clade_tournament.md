# Clade tournaments: ranking agents that do the same job

A clade tournament takes one branch of the agent tree (agents that claim similar jobs) and ranks
its members on a shared test battery, run under identical conditions (a common garden). First
target: the debugging clade (18 agents from 5 repos, including the documented copy
`lst97/debugger` of `wshobson/debugging-toolkit-debugger`).

## Stages

1. **Prepare** (built; `phailogeny tournament prepare`)
   - Freeze the clade's members from the tree (`members.json`).
   - Claim matrix: a model reads every member's description and prompt excerpt and lists the
     capabilities the clade claims and which member claims which (`claims.json`). Tests come from
     this consensus, so no member's own wording seeds them.
   - Generate tasks per capability: verifiable (planted bug + pytest), rubric, disposition,
     robustness (`tasks.yaml`).
   - Validate locally, no model calls: the buggy code must fail its test and the reference fix
     must pass (`validation.json`).
   - Add hold-out tasks from an independent source (`data/probes.yaml`).
   - **A human reviews `tasks.yaml` before any agent runs.**
2. **Run** (next): every member plus two controls (no system prompt; a generic good prompt), same
   model and sandbox, 3 replicates, randomised order. Verifiable answers are scored by running the
   tests on the agent's patch.
3. **Score and rank** (next): pass rates for verifiable tasks; blind, order-swapped pairwise judging
   for rubric tasks (top members only), fitted with Bradley-Terry; bootstrap over tasks for rank
   confidence intervals; per-capability leaderboards and a quality-vs-cost Pareto front; uplift
   over the no-prompt baseline.
4. **Decide**: promote, merge best sections, retire dominated agents, flag unsupported claims.

## Results: first run, verifiable tasks only (2026-10-01)

18 debugging agents + 2 controls (no system prompt; a generic expert prompt) × 26 verifiable
bug-fix tasks × 1 replicate, Claude Opus 5.5 at low effort, patches tested in a macOS sandbox
(no network, no writes outside a temp folder). Cost: $21.43 at list price.

- **All 520 patches pass**, the no-prompt control included: a full ceiling. These tasks are too
  easy to separate agents on this model.
- **Agents cost more for the same result.** Paired over the same tasks (ratio of total costs),
  all 18 agents cost significantly more than no prompt: 10-42% more, median 27%, because their
  prompts lengthen answers. (An earlier mean-of-per-task-ratios figure, +30%, was skewed by tasks
  where the control was very cheap.)
- Scoring fixes made along the way (stored answers were re-scored offline): the extractor now
  picks the block that defines the module (agents often append a test block) and accepts
  four-backtick fences. Both bugs had marked correct patches as failures.

Plots: `out/figures/tournament_debugging_paired_cost.svg` (the paired test) and
`out/figures/tournament_debugging.svg` (absolute cost); numbers:
`data/tournaments/debugging/ranking_claude-opus-5-5.json`.

## Results: Haiku 4.5 rerun (2026-10-01)

Same agents, tasks and sandbox on Claude Haiku 4.5 (low effort), to push agents off the ceiling.
Cost: $9.96 at list price.

- **511 of 520 patches pass.** Both controls fix all 26 bugs. The 9 misses (audited: all genuine
  wrong fixes, none extraction errors) are spread over 8 agents and concentrated in 2 tasks.
- **No reliable ranking.** Every agent's 95% Wilson interval overlaps 100% (26/26 gives
  0.87-1.00; 25/26 gives 0.81-0.99). Agents vs controls, 9/468 vs 0/52 failures: Fisher p = 0.61.
  (Pass-rate intervals were first drawn as percentile bootstraps, which collapse to zero width at
  26/26 and overstate certainty; ranking files now carry Wilson intervals too.)
- **Cost:** Haiku answers cost about half of Opus answers (1.9 vs 4.1 cents per task). On Haiku,
  agent prompts barely change cost (median +2%; 2 of 18 significant).
- **Reading:** the model decision moves cost far more than the agent decision moves quality, and
  ranking these agents needs harder tasks.

Plots: `out/figures/slide_fig5_haiku_pass_wilson.svg` (Wilson intervals),
`out/figures/tournament_debugging_paired_cost_haiku.svg`; Opus cost: `out/figures/slide_fig4_opus_extra_cost.svg`.
Next: harder tasks (multi-file, subtle, under-specified), and the rubric and disposition tasks,
where a prompt's rules should matter more than its knowledge.

## Results: harder multi-file tasks, Opus vs Haiku (2026-10-01)

12 generated tasks (`data/tournaments/debugging-hard/`): 3-5 modules, 96-154 lines, a bug that
spans modules with the symptom far from the cause, plus a red herring; all 12 validated (bug fails
the hidden tests, reference fix passes). Two agents (wshobson and VoltAgent debuggers) + both
controls, 1 replicate per model. Generation $2.17, runs $3.32.

| Model | Patches passing | Mean cost per task | Cost per correct fix |
| --- | --- | --- | --- |
| Opus 5.5 | 48 / 48 | 4.2 cents | 4.2 cents |
| Haiku 4.5 | 45 / 48 | 2.7 cents | 2.9 cents |

Controls: on Opus both fixed all 12; on Haiku both missed the same one task as VoltAgent's
debugger (11 / 12), and only wshobson's debugger fixed all 12, too little to call it better.

- Opus still solves everything; these "hard" tasks are not hard enough to find its limit.
- All three Haiku failures are one task, and the same mistake: it found the right file but returned
  it incomplete, silently dropping a function. A sloppy edit, not a wrong diagnosis.
- Across both task sets, all 12 tasks where the models disagreed went Opus's way (sign test
  p = 0.0005): Opus is reliably better, by a few percentage points, at roughly twice the cost.
- Scorer fix (re-scored offline, 1 verdict changed): when an answer labels its files, unlabelled
  illustrative snippets are no longer allowed to overwrite a file.

## Future test suites (logged, not built)

- **Generalisability to a different task.** Run each agent on another clade's battery (e.g.
  debuggers on code-review tasks) to measure specialist vs generalist. The full design is a
  transfer matrix: clades × batteries, each cell the drop from home performance. Biology analogue:
  a reciprocal transplant experiment, where populations are grown in each other's environments
  to separate local adaptation from general vigour.
