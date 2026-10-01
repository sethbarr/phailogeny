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
- **No reliable ranking.** Every agent's pass-rate interval overlaps 100%; rank intervals span
  most of the field. Agents vs controls, 9/468 vs 0/52 failures: Fisher p = 0.61.
- **Cost:** Haiku answers cost about half of Opus answers (1.9 vs 4.1 cents per task). On Haiku,
  agent prompts barely change cost (median +2%; 2 of 18 significant).
- **Reading:** the model decision moves cost far more than the agent decision moves quality, and
  ranking these agents needs harder tasks.

Plots: `out/figures/tournament_debugging_haiku.svg`, `out/figures/tournament_debugging_paired_cost_haiku.svg`.
Next: harder tasks (multi-file, subtle, under-specified), and the rubric and disposition tasks,
where a prompt's rules should matter more than its knowledge.

## Future test suites (logged, not built)

- **Generalisability to a different task.** Run each agent on another clade's battery (e.g.
  debuggers on code-review tasks) to measure specialist vs generalist. The full design is a
  transfer matrix: clades × batteries, each cell the drop from home performance. Biology analogue:
  a reciprocal transplant experiment, where populations are grown in each other's environments
  to separate local adaptation from general vigour.
