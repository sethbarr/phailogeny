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

## Future test suites (logged, not built)

- **Generalisability to a different task.** Run each agent on another clade's battery (e.g.
  debuggers on code-review tasks) to measure specialist vs generalist. The full design is a
  transfer matrix: clades × batteries, each cell the drop from home performance. Biology analogue:
  a reciprocal transplant experiment, where populations are grown in each other's environments
  to separate local adaptation from general vigour.
