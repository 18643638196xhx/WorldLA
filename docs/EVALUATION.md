# RoboTwin integration

The simulator, robot assets, planning libraries, Vulkan/SAPIEN dependencies,
and accepted seed/instruction caches must be installed separately. This
project does not modify simulator assets, discard difficult seeds, or remove
clutter objects. Such changes alter evaluation and must not be hidden.

`worldla.integrations.robotwin` provides the `get_model(args)`,
`eval(task_env, model, observation)`, and `reset_model(model)` policy interface.
It expects the observation keys and joint `take_action()` convention used
by the compatible RoboTwin evaluator. It uses only the exported full model;
it does not reconstruct absolute paths to a training checkpoint.

Example, from a shell with WorldLA installed and one GPU made visible:

```bash
CUDA_VISIBLE_DEVICES=0 worldla eval-robotwin \
  --robotwin-root /path/to/compatible/RoboTwin \
  --eval-script script/eval_policy.py \
  --deploy-config configs/deploy_robotwin.yaml \
  --model-dir checkpoints/worldla-100k/original \
  --task open_microwave --task-config demo_clean --episodes 100 \
  --accepted-seed-dir data/accepted_seeds \
  --summary outputs/eval/open_microwave.jsonl --dry-run
```

Inspect the generated command, then remove `--dry-run` to run it. The child
process registers the WorldLA plug-in by module name before running the
external evaluator. This does not vendor or certify every RoboTwin/XPolicyLab
release. An evaluator with a different CLI or policy import contract needs a
corresponding adapter. A missing evaluator/asset is an error, not a successful
evaluation. No full simulator rollout has been certified merely by dry-run tests.

For paired comparisons, hold fixed task configuration, accepted seeds and
instructions, simulator/assets version, checkpoint, sampling seed, denoising
steps, action horizon, execution interval, and gripper handling. Report clean
and randomized separately. Never average partial jobs as if all tasks finished.

The RoboTwin plug-in resets the sampling key at each episode using
`inference_seed + 1009 * episode_index`, starting at episode index zero,
matching the reference evaluator. Standalone `policy.reset()` resets to its
initial seed unless an explicit seed is supplied. Do not conflate these two uses.

Default old-plan arm weight is 0.75. Use `--old-plan-weight 0` for all-new
replanning. Gripper actions switch discretely at new weight 0.5. Each new
plan still uses current observations, even if old arm weight is 1.0.
JAX device selection and Vulkan rendering selection are separate concerns;
the launcher binds SAPIEN 3's renderer and scene to the first visible CUDA
device. Verify the printed physical PCI address before multi-GPU runs.
Older SAPIEN versions without the Device API retain native renderer selection.

## Isolated closed-loop smoke test

When official simulator/assets and a compatible legacy evaluation driver live
in separate checkouts, `scripts/smoke_evaluation.py` creates a fresh compatibility
root using symlinks. It does not edit either checkout. It runs the installed
WorldLA package, not a policy implementation from the external evaluator.
The output directory must not exist; failed attempts are preserved.

```bash
python scripts/smoke_evaluation.py \
  --robotwin-root /path/to/official/RoboTwin \
  --eval-script /path/to/compatible/RoboTwin/script/eval_policy.py \
  --deploy-config configs/deploy_robotwin.yaml \
  --original checkpoints/worldla-100k/original \
  --merged checkpoints/worldla-100k/merged \
  --seed-cache data/accepted_seeds/press_stapler__demo_clean__seedgroup0.json \
  --task press_stapler --gpu 0 --episodes 3 \
  --output outputs/smoke_press_stapler
```

The helper copies the first requested accepted seeds/instructions into the
new output directory and runs the two checkpoint formats sequentially on the
same physical GPU. It refuses to start on a GPU with 512 MiB or more in use.
Each run must exit successfully and produce a structured summary with the
requested episode count. A zero exit status from a renderer self-test alone
is not sufficient. Logs record the policy/model/backend import locations and
renderer device. This small test checks the evaluation pipeline; it does not
estimate the 50-task success rate or establish equivalence of the two formats.
