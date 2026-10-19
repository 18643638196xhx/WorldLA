# WorldLA

Stage-conditioned action–visual relation transfer from Fast-WAM to pi0.5,
with time-aligned old-tail/new-head plan fusion.

[中文说明](README_zh.md) · [Training and data](docs/TRAINING.md) ·
[Evaluation](docs/EVALUATION.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

Validation: **30 CPU tests passed**; the independently installed package exactly
matches saved reference actions/routes/plans on four real fixed observations.
Both full-checkpoint formats also completed a real `press_stapler/demo_clean`
closed-loop smoke test (3/3 successes each, same seeds). This is not a new
training or 50-task benchmark. [Validation details](docs/VALIDATION.md).

## What this repository contains

- **What to transfer:** offline shared-K spatial reading distributions,
  aggregated by relative action-chunk position. Teacher visual values and
  teacher action outputs are not the main supervision targets.
- **How to learn:** a context-adaptive stage router predicts distributions
  over pi0.5's own visual tokens. A message adapter/gates inject the routed
  student visual content into corresponding action tokens during denoising.
- **How to execute:** fuse the time-aligned previous plan tail and new plan
  head; default old/new arm weights are 0.75/0.25.

Fast-WAM is needed only to construct teacher caches, not in student training
from caches or deployed inference. Four relative chunk stages are **not**
automatically the physical phases approach/grasp/transport/place.

This is a standalone source project: it does not import an old experiments
checkout or depend on development-server paths. The tested OpenPI snapshot
is embedded, with its license and provenance. Use a clean environment rather
than installing over another active OpenPI environment.

## Install

Use Linux and Python **3.11**. From the repository root:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
worldla --help
```

The tested requirements include CUDA 12 JAX; a compatible NVIDIA driver is
required. For training, additionally install `requirements-train.txt`. Never
upgrade an environment running other training/evaluation jobs.

## Download and run an existing checkpoint

The model repository is separate from this code project. It may be private
or still uploading; authenticate locally if needed and wait until all weight
files are available before using it:

```bash
huggingface-cli login
huggingface-cli download xhx0327/WorldLA-100k-decay-full \
  --include 'original/*' --local-dir checkpoints/worldla-100k
CUDA_VISIBLE_DEVICES=0 worldla infer \
  --model-dir checkpoints/worldla-100k/original --dummy
```

`--dummy` is a loading/shape check only. **Never execute its outputs on a
real robot.** For actual observations, use `--input observation.npz` with
`state` (14), and RGB uint8 CHW `cam_high`, `cam_left_wrist`, `cam_right_wrist`.

```python
from worldla import WorldLAPolicy

policy = WorldLAPolicy('checkpoints/worldla-100k/original')
policy.reset()  # Every episode.
chunk = policy.infer(observation)['actions']  # (50,14), fresh chunk, no fusion.
commands = policy.plan(observation)          # (10,14), cached-tail fusion.
```

The observation also contains a string `prompt`, with images nested in an
`images` dictionary. State/actions are RoboTwin ALOHA joint coordinates, not
Cartesian commands. Execute all ten commands before the next `plan()` call.
Grippers switch discretely. Sampling uses ten denoising steps.

Both `original/` and `merged/` full-checkpoint formats are supported. Prefer
`original/` for reproduction. LoRA folding changes low-precision arithmetic;
fixed-input agreement does not establish identical rollout success rates.

## Train

Prepare a local LeRobot v2.1 dataset, matching normalization assets, a pi0.5
base checkpoint, tokenizer, and cached stage routes. See `docs/TRAINING.md`.
All paths in example configs are relative to the working directory and can
be overridden with command-line flags:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 worldla train --config configs/router.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 worldla train --config configs/action_100k.yaml
```

These commands are examples, not automatically started by installation or tests.
The second stage freezes the router and trains LoRA in the VLM/action expert
plus the message adapter/gates. Parameter-only warm starts reset optimizer state;
they are not seamless training resumes. No full training run is claimed
for this refactor until it has been separately validated.

## Export and evaluate

```bash
worldla export --model-dir outputs/action_100k --variant original \
  --tokenizer data/paligemma_tokenizer.model --output-dir outputs/release-original
worldla export --model-dir outputs/action_100k --variant merged \
  --tokenizer data/paligemma_tokenizer.model --output-dir outputs/release-merged
```

Exports refuse to overwrite or overlap source checkpoints and verify parameter
round trips/source fingerprints. They do not automatically rerun task success.
See `docs/EVALUATION.md` for the RoboTwin adapter and external evaluator command.

## Scope and release status

Core inference, cached-route training, route preparation, export, and a simulator
adapter are included. The historical Fast-WAM discovery/instrumentation stack
and raw simulator/data assets are **not** silently copied here. Raw teacher
extraction needs a separately integrated, validated Fast-WAM adapter; the exact
required cache format is documented. This is a first source-project release,
not a claim that every training/simulator configuration has been rerun.

No unverified benchmark table, invented paper link, or fabricated BibTeX is
included. Add the finalized paper/code links and version-matched results
before public announcement.

## License

Apache-2.0 for the project's code. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`. Model weights, tokenizers, datasets, and simulator
assets have separate terms; this license does not relicense them.
