# Validation recorded on 2026-10-20

These checks validate this source refactor; they are not new benchmark results.

## WorldLA 0.2 namespace migration

The namespace/file/class migration was tested separately from the initial
standalone release. Existing checkpoint directories and parameter keys were
not rewritten.

| Check after renaming | Result |
| --- | --- |
| CPU tests, source and relocated installation | 32 passed in each |
| Wheel completeness | All 107 Python modules included; no `openpi` or `openpi_client` package shipped |
| Public model API | `WorldLAConfig`, `WorldLABackbone`, `WorldLAModel` |
| Existing checkpoint layout | All 89 parameter paths and shapes match the recorded pre-migration fixture |
| Original-format checkpoint, four real observations | Actions, normalized actions, routes and replanned commands exactly match its own pre-migration reference |
| Merged-format checkpoint, same four observations | The same four output groups exactly match its own pre-migration merged reference |
| Fixed-input RMSE / maximum absolute difference | 0 / 0 for both formats |
| Real-data two-stage training | Four updates per stage, four FSDP GPUs; checkpoint save and restore passed |
| Training regression | All eight training losses and both validation metric dictionaries exactly match the pre-migration smoke run |
| Action-stage parameter checks | 20 LoRA and 5 adapter/gate leaves updated; all 11 Router leaves frozen |
| Post-migration simulator rollout, original format | `press_stapler/demo_clean`, three fixed seeds: 3/3 successes, exit 0 |
| Post-migration simulator rollout, merged format | Same three fixed seeds: 3/3 successes, exit 0 |

The two checkpoint formats were each compared against **their own** saved
outputs; this does not assert equality between original and merged arithmetic.
Detailed local records (ignored by Git) are in
`outputs/namespace_verification_20261020/` and
`outputs/namespace_train_smoke_20261020_01/status.json`.
The installed modules resolve under `worldla`; no legacy namespace is loaded
in the model/API tests and checkpoint verification.

Both post-migration rollouts used physical GPU 6, ten denoising steps, a
50-action horizon, execution interval 10, and old-plan arm weight 0.75. Seeds
were 100000, 100002 and 100003. The policy/model paths in both logs resolve
inside the renamed installed package. Records are
`outputs/namespace_eval_smoke_20261020_01/status.json`, the two `.log` files,
and the two `.summary.jsonl` files. These are pipeline smoke tests on one task,
not new task-success benchmarks. The isolated training dataset cache was
removed after verification; logs and smoke checkpoints were retained.

These checks reused the existing dependency environment. Four updates per
stage validate the training path, not full-budget convergence. The optional
compatibility architectures and Torch backend are not covered by these JAX
deployment checks. See `NAMESPACE_MIGRATION.md` for API and provenance details.

## Initial standalone release (WorldLA 0.1.0)

The records below predate the 0.2 namespace migration. In particular, the
source-hash equality below refers to the original standalone snapshot, not to
the renamed files.

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
| Real-data Router training smoke test | Four FSDP GPUs, four updates, finite loss/nonzero gradients, validation and checkpoint save passed |
| Real-data action training smoke test | Four updates, intermediate/final checkpoint save and CPU restore passed |
| Action-stage parameter updates | All 20 LoRA leaves and all 5 message-adapter/gate leaves changed between updates 2 and 4 |
| Action-stage Router freeze | All 11 Router leaves unchanged and equal to the pretrained Router after BF16 conversion |

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

## Two-stage training smoke evaluation

The installed package, outside the source checkout, trained on the actual
local 548,893-frame LeRobot dataset and existing 20,000-anchor teacher cache.
Physical GPUs 0--3 were used with FSDP=4, global batch 4, and seed 20260901.
Router training used 16,000 train anchors; action training indexed all 439,140
frames belonging to train episodes. The held-out validation pool contained
2,000 anchors; only one four-sample validation batch was evaluated per stage.

Both stages completed four updates and exited 0. Router training losses were
0.369038, 0.282671, 0.152752 and 0.172069; action losses were 0.140943,
0.099055, 0.110947 and 0.055456. Final smoke validation KL/action loss were
0.131471/0.104603, respectively. These values use tiny batches and are **not**
full-budget validation scores or evidence of convergence.

The action stage restored the freshly trained Router, applied the scheduled
learning-rate code, saved step-2 and final step-4 adaptation checkpoints, then
restored their parameters on CPU. Every LoRA and message-adapter/gate leaf
changed between the snapshots. Router leaves were exactly unchanged, and
matched the separate Router checkpoint after its documented BF16 conversion.
No dense-base parameters were written into the adaptation checkpoint.

Local records are `outputs/train_smoke_20261020_01/status.json`, `router.log`,
`action.log`, and the per-stage metric/checkpoint subdirectories. The test used
the existing dependency environment without upgrading it; it did not verify a
fresh dependency installation or a complete 60k/100k training run. No model-core
changes were needed to make this smoke test pass.

Not yet validated by these checks:

- Full-budget Router/action convergence and reproduction of trained success rates after the refactor.
- A full 50-task success-rate benchmark, randomized rollout, or public leaderboard submission.
- Fresh raw Fast-WAM discovery/extraction in a separately installed teacher environment.
- Every other Python/CUDA/hardware combination or every RoboTwin evaluator version.

The CPU tests emit one harmless warning in the read-only shared test environment:
numpydantic cannot generate an optional type stub in that environment. The
existing environment was not upgraded or modified to suppress the warning.
