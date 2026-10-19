# Validation recorded on 2026-10-20

These checks validate this source refactor; they are not new benchmark results.

| Check | Result |
| --- | --- |
| CPU unit tests | 30 passed |
| Wheel completeness | All 105 Python source modules included |
| Relocated installation | Installed into a fresh temporary directory, not the development environment |
| CLI interfaces | All six subcommands expose help; RoboTwin command generation dry-run checked |
| Core implementation | WorldLA model and pi0/Gemma/LoRA core source hashes match the prior implementation |
| Existing original-graph checkpoint | Loaded from the independently installed package |
| Four real fixed observations | Normalized actions, joint actions, routes and cached-tail plans exactly matched the saved reference |
| RMSE / maximum absolute difference | 0 / 0 for all four output groups |
| Closed-loop smoke evaluation, original format | `press_stapler`, `demo_clean`: 3/3 successes, process exit 0 |
| Closed-loop smoke evaluation, merged format | Same task/seeds: 3/3 successes, process exit 0 |

The four observations are the Approach/Grasp/Transport/Place states from one
previously captured trajectory. The reference inference uses seed 20260901,
ten denoising steps, and the original-graph scheduled-decay 100k checkpoint.
Model imports were confirmed to resolve to the independently installed package,
not the original project or the checkpoint folder's vendored Python files.

The local detailed report is generated at
`outputs/verification/installed_package.json`; it is ignored by Git because
it contains machine-specific paths. Private observations and weights are not
bundled in this repository or wheel.

## Closed-loop simulator smoke evaluation

The independently installed wheel was run outside the source tree on one
A100 80 GB (physical GPU 6). Both full-checkpoint formats completed three
episodes using the same accepted seeds (100000, 100002, 100003) and cached
seen instructions. Settings were `demo_clean`, inference seed 20260901,
ten denoising steps, a 50-action horizon, execution interval 10, and old-plan
arm weight 0.75. Episode sampling resets followed the convention below in
`EVALUATION.md`.

The external compatible evaluation driver used official simulator/assets
through an isolated symlink root. Neither simulator assets nor source
checkpoints were changed. Renderer logs confirm PCI `0000:da:00.0`, and
policy/model/OpenPI module paths resolve inside the temporary installed
package, not the historical WorldLA experiment implementation. Integration
fixes included required evaluator identifiers, compatibility-root path
preservation, external render-check helper linking, and SAPIEN device binding.

Both runs exited 0, wrote `_result.txt`, and produced structured summaries
with three trials and three successes. Local records (ignored by Git):

- `outputs/eval_smoke_20261020_02/status.json`
- `outputs/eval_smoke_20261020_02/original.summary.jsonl`
- `outputs/eval_smoke_20261020_02/merged.summary.jsonl`
- `outputs/eval_smoke_20261020_02/original.log`
- `outputs/eval_smoke_20261020_02/merged.log`

This validates the evaluation pipeline on one task, not a 50-task success
rate, randomized evaluation, or statistical equivalence of the two formats.

Not yet validated by these checks:

- A new end-to-end router/action training run after the refactor.
- A full 50-task success-rate benchmark, randomized rollout, or public leaderboard submission.
- Fresh raw Fast-WAM discovery/extraction in a separately installed teacher environment.
- Every other Python/CUDA/hardware combination or every RoboTwin evaluator version.

The CPU tests emit one harmless warning in the read-only shared test environment:
numpydantic cannot generate an optional type stub in that environment. The
existing environment was not upgraded or modified to suppress the warning.
