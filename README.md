# WorldLA

Transfer action–visual relations from Fast-WAM to an efficient pi0.5 policy,
with time-aligned old-tail/new-head replanning.

[Paper (PDF)](docs/WorldLA.pdf) · [中文完整流程](README_zh.md) ·
[Hugging Face release](https://huggingface.co/xhx0327/WorldLA-100k-decay-full) ·
[Training details](docs/TRAINING.md) · [Evaluation details](docs/EVALUATION.md) ·
[Validation](docs/VALIDATION.md)

## Choose a workflow

| Goal | Required materials | Start here |
| --- | --- | --- |
| Load a trained policy | This code + one full checkpoint | Sections 1–2 |
| Evaluate in RoboTwin | Full checkpoint + compatible simulator/evaluator/assets | Sections 1–3 |
| Retrain from cached supervision | 20K routes + matching robot demonstrations + pretrained pi0.5 | Sections 1, 4–7 |
| Export a trained policy | Completed action run + its initialization/assets | Section 8, then section 3 |

Fast-WAM runs **only during offline teacher construction**, not when training
from caches or deploying WorldLA. The router predicts **where to read**; visual
content comes from the student's own tokens. The four training stages are
relative action-chunk bins, **not** automatically recognized physical
approach/grasp/transport/place phases.

## 1. Install

Use **Linux, Python 3.11 and a CUDA-12-compatible NVIDIA driver**. Create a new
environment; do not upgrade another running robotics experiment's environment.
Commands use **Bash**, remain in the repository root, and use absolute
environment-variable paths. Re-export these variables in a new terminal.

```bash
git clone https://github.com/18643638196xhx/WorldLA.git
cd WorldLA
export WORLDLA_PROJECT_ROOT="$(pwd)"
python3.11 -m venv .venv
. .venv/bin/activate
set -euo pipefail

# Inference/evaluation dependencies:
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
worldla --help
```

For **training**, use `requirements-train.txt` instead of the runtime-only
file above; it includes the runtime requirements. If runtime is already
installed, add:

```bash
python -m pip install -r requirements-train.txt
python -m pip install h5py
```

Pins include JAX 0.5.0, Flax 0.10.2, NumPy 1.26.4, and LeRobot commit
`a445d9c9da6bea99a8972daa4fe1fdd053d711d2`. Do not substitute an unrelated
latest LeRobot version. Simulator dependencies are installed separately.

### Hugging Face paths and revision

The model repository is **public**; login is not required. All examples pin
the release containing both weights and route data:

```bash
export WORLDLA_HF_REPO=xhx0327/WorldLA-100k-decay-full
export WORLDLA_HF_REVISION=7fd1111c6fa8419a64bb6f26383b5025ffbf436d
export WORLDLA_RELEASE_ROOT="$WORLDLA_PROJECT_ROOT/checkpoints/worldla-100k"
```

The code lineage recorded by the release is
`31301d2c33d90b86afab0e0a5168b780d24378c6`; see the bundle's `provenance.json`.
Use `huggingface-cli login` only if your download environment requires it;
never put an access token in a command, README, or issue.

## 2. Download a checkpoint and test inference

Choose **one** full-model folder; neither needs a separate base checkpoint or
Fast-WAM at inference time.

| Hub folder | Approximate size | Contents / use |
| --- | --- | --- |
| `original/` | 5 GiB | Full base + separate LoRA + router + adapter/gates + tokenizer/stats; preferred for reproduction |
| `merged/` | 9.5 GiB | LoRA folded into FP32 dense weights; router/adapter still explicit |
| `training/stage_routes_20k/` | 96.3 MiB archive | Teacher routes, configs, normalization, alignment metadata and tools; no robot images/actions |

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'original/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_MODEL_DIR="$WORLDLA_RELEASE_ROOT/original"

# Loading/shape check only; executes no simulator or robot commands.
CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" --dummy
```

To use the folded version, download `'merged/*'` with the same command and
set `WORLDLA_MODEL_DIR="$WORLDLA_RELEASE_ROOT/merged"`. These are custom
Flax/Orbax full checkpoints, not Transformers `AutoModel` models. WorldLA
loads them through the installed package, not the Python files bundled with
the weights. Folding changes low-precision arithmetic; do not assume identical
rollout success rates between formats.

For a **real observation**, prepare an NPZ with `state` (float32, `[14]`) and
`cam_high`, `cam_left_wrist`, `cam_right_wrist` (RGB uint8, `[3,H,W]`).
Use the corresponding ALOHA joint/gripper convention, not Cartesian poses:

```bash
CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" \
  --input /absolute/path/to/observation.npz \
  --prompt 'Place the can in the basket.' \
  --output "$WORLDLA_PROJECT_ROOT/outputs/inference/actions.npz"
```

The output must be a finite `[50,14]` action chunk. Existing output files are
refused. The first call includes model loading/JIT and is not a steady-state
latency measurement. **Never execute dummy-input outputs on a robot.**

Python interface (run with a free GPU visible):

```python
import os
import numpy as np
from worldla import WorldLAPolicy

with np.load('/absolute/path/to/observation.npz', allow_pickle=False) as data:
    observation = {
        'images': {k: data[k] for k in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
        'state': data['state'],
        'prompt': 'Place the can in the basket.',
    }
policy = WorldLAPolicy(os.environ['WORLDLA_MODEL_DIR'])
policy.reset()  # Every episode: clears the old plan and resets sampling.
commands = policy.plan(observation)  # [10,14], time-aligned fused commands.
# Control loop: execute ALL 10 commands, acquire a NEW observation, then plan().
# This example does not execute commands.
```

`infer(observation)['actions']` returns a fresh 50-step chunk without fusion.
Default deployment uses 10 denoising steps, 10 executed actions per replan,
and old/new **arm** weights 0.75/0.25. Grippers switch discretely.

## 3. Evaluate in RoboTwin

Install RoboTwin, SAPIEN/Vulkan, robot/planner dependencies and **all assets**
separately, following the [upstream project](https://github.com/RoboTwin-Platform/RoboTwin).
The evaluation Python environment must import both WorldLA and the simulator.

**Compatibility boundary:** this project supplies the legacy
`get_model / eval / reset_model` adapter and a launcher for a compatible
`script/eval_policy.py`. It does **not** yet adapt every current XPolicyLab
interface. Cloning latest RoboTwin alone does not guarantee evaluation-CLI
compatibility. A different interface needs adaptation; see the
[evaluator contract and paired smoke test](docs/EVALUATION.md).

Point to your **already installed compatible evaluation checkout**, then
inspect the command without starting a rollout:

```bash
export WORLDLA_ROBOTWIN_ROOT=/absolute/path/to/compatible/RoboTwin
mkdir -p "$WORLDLA_PROJECT_ROOT/outputs/eval"

CUDA_VISIBLE_DEVICES=0 worldla eval-robotwin \
  --robotwin-root "$WORLDLA_ROBOTWIN_ROOT" \
  --eval-script script/eval_policy.py \
  --deploy-config "$WORLDLA_PROJECT_ROOT/configs/deploy_robotwin.yaml" \
  --model-dir "$WORLDLA_MODEL_DIR" --task press_stapler \
  --task-config demo_clean --episodes 3 --seed-group 0 \
  --inference-seed 20260901 --old-plan-weight 0.75 \
  --run-tag worldla_smoke \
  --summary "$WORLDLA_PROJECT_ROOT/outputs/eval/press_stapler_smoke.jsonl" \
  --dry-run
```

Remove `--dry-run` to run the three-episode smoke test. For task evaluation,
use `--episodes 100`, a new `--run-tag` and a new summary path. Change `--task`
for other tasks; use `--task-config demo_randomized` for randomized evaluation
(unseen instructions; clean uses seen instructions). Keep clean/randomized
results separate. Three smoke episodes are not a benchmark SR estimate.

For paired reproduction, also pass
`--accepted-seed-dir /absolute/path/to/matching/accepted_seeds` to a compatible
driver. These simulator seed/instruction caches are **not** the 20K training
route cache and are not included in it. Without them, do not claim exact
historical paired-seed results. Log versions, assets, instructions and seeds;
never silently skip hard seeds, remove clutter, or count environment crashes
as policy failures. `--old-plan-weight 0` disables old-plan fusion.

Append `2>&1 | tee outputs/eval/press_stapler.log` to save logs; from another
terminal use `tail -f outputs/eval/press_stapler.log`. Inspect the printed
renderer physical PCI device: CUDA and Vulkan device selection are separate.
A renderer error or a successful dry-run is **not** a completed evaluation.
`scripts/smoke_evaluation.py` can check both formats on identical accepted
seeds; its full command is in the evaluation guide.

## 4. Download the training bundle

**Skip sections 4–7 for inference/evaluation only.** Retraining needs three
distinct inputs: published route bundle, matching robot demonstrations, and
the **pretrained** pi0.5 initialization. A fine-tuned full WorldLA checkpoint
is not the initialization used by this training recipe.

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'training/stage_routes_20k/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"

# Verify before extraction, against the fixed release checksum.
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['WORLDLA_RELEASE_ROOT']) / 'training/stage_routes_20k'
meta = json.loads((root / 'archive_sha256.json').read_text())
archive = root / meta['filename']
expected = '43ab0e7cb9d2fec12813a351bedb9957af1847cb2a1601c68af5e1f9a5a412dc'
assert archive.stat().st_size == meta['bytes'] == 100959563
assert hashlib.sha256(archive.read_bytes()).hexdigest() == meta['sha256'] == expected
print('Training archive checksum OK')
PY
mkdir -p "$WORLDLA_PROJECT_ROOT/data"
# Use a new directory, not an existing/modified training bundle.
test ! -e "$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
tar -xzf "$WORLDLA_RELEASE_ROOT/training/stage_routes_20k/stage_routes_20k.tar.gz" \
  -C "$WORLDLA_PROJECT_ROOT/data"
export WORLDLA_REPRO_ROOT="$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
JAX_PLATFORMS=cpu python "$WORLDLA_REPRO_ROOT/tools/verify_bundle.py" \
  --root "$WORLDLA_REPRO_ROOT" --worldla-loader
```

The archive includes `samples/`, `stage_route_manifest.jsonl`, train-only
`train_stage_priors.npz`, matching `assets/`, dataset-alignment fingerprints,
`configs/`, `tools/`, checksums and provenance. Expected verification:
**20,000 anchors**, **2,500 episodes**, **16,000/2,000/2,000 train/val/test**,
and targets of shape **[4,768]**. Splits are episode-disjoint.

These are 20K **derived frame anchors**, not 20K additional demonstrations.
The action stage uses **all 439,140 frames from train episodes**, not just
16K route anchors. No Fast-WAM download is needed for this path.

## 5. Obtain and align the robot dataset

Robot images/actions are **not** in the route bundle. Download the 50-task,
50-demonstration-per-task `demo_clean.zip` collection from
[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).
Use **`dataset/<task>/demo_clean.zip`**, not the randomized-500 collection.

Use a separate data-preparation checkout, not the evaluation checkout from section 3:

```bash
mkdir -p "$WORLDLA_PROJECT_ROOT/third_party"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git \
  "$WORLDLA_PROJECT_ROOT/third_party/RoboTwin-data"
export WORLDLA_DATA_TOOLS="$WORLDLA_PROJECT_ROOT/third_party/RoboTwin-data"
git -C "$WORLDLA_DATA_TOOLS" checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
git -C "$WORLDLA_DATA_TOOLS" submodule update --init XPolicyLab
git -C "$WORLDLA_DATA_TOOLS/XPolicyLab" checkout 0ccd8e9f3ed76c1a3186a5b151bead5b53cf2249

export WORLDLA_RAW_ROOT="$WORLDLA_PROJECT_ROOT/data/robotwin_raw"
ROBOTWIN_DATA_ROOT="$WORLDLA_RAW_ROOT" \
HF_ARCHIVE_CACHE="$WORLDLA_PROJECT_ROOT/data/robotwin_archives" \
HF_ARCHIVE_NAME=demo_clean.zip HF_MAX_WORKERS=4 HF_EXTRACT_WORKERS=4 \
  bash "$WORLDLA_DATA_TOOLS/scripts/download_xpolicylab_data.sh"

export WORLDLA_LEROBOT_ROOT="$WORLDLA_PROJECT_ROOT/data/lerobot"
python "$WORLDLA_REPRO_ROOT/tools/audit_official_data.py" \
  --data-root "$WORLDLA_RAW_ROOT" \
  --task-list "$WORLDLA_REPRO_ROOT/dataset_alignment/task_order.yaml" \
  --output "$WORLDLA_PROJECT_ROOT/data/official_inventory.jsonl"

python "$WORLDLA_REPRO_ROOT/tools/convert_official_lerobot.py" \
  --inventory "$WORLDLA_PROJECT_ROOT/data/official_inventory.jsonl" \
  --dataset-root "$WORLDLA_LEROBOT_ROOT/robotwin_official50_demo_clean_v21" \
  --repo-id robotwin_official50_demo_clean_v21 \
  --xpolicylab-parent "$WORLDLA_DATA_TOOLS" --seed 20261004

JAX_PLATFORMS=cpu python "$WORLDLA_REPRO_ROOT/tools/verify_bundle.py" \
  --root "$WORLDLA_REPRO_ROOT" \
  --dataset-root "$WORLDLA_LEROBOT_ROOT/robotwin_official50_demo_clean_v21" \
  --worldla-loader
```

Native raw layout: `<WORLDLA_RAW_ROOT>/demo_clean/<task>/aloha_agilex/{data,instruction}/`.
`WORLDLA_LEROBOT_ROOT` is the **parent** of the dataset directory. Keep fixed
task/episode/frame order, seen-instruction selection, RGB decoding and joint
conventions. Do not swap channels or substitute an arbitrary LeRobot conversion.
The converter supports `--resume` for interruption, not overwriting from scratch.

The verifier checks all 2,500 episodes' prompts, frame order and numeric
state/action fingerprints; expected source size is **548,893 frames**. It does
not hash every image pixel. The historical upstream dataset Hub revision was
not recorded, so the alignment check is mandatory; **stop if it fails** rather
than training against mismatched targets. Reserve disk for ZIPs, extracted
data, the converted dataset and runtime caches, not just the 96 MiB route file.

## 6. Download initialization and set training paths

Use the JAX pi0.5 base published by
[Physical Intelligence](https://github.com/Physical-Intelligence/openpi):
`gs://openpi-assets/checkpoints/pi05_base`. An unrelated converted PyTorch
checkpoint is not interchangeable. If already downloaded, skip the GCS command
and point `WORLDLA_INIT_PARAMS` at its actual `params/` directory.

```bash
export WORLDLA_DATA_HOME="$WORLDLA_PROJECT_ROOT/data/pretrained_cache"
python - <<'PY'
from worldla.utils.downloads import maybe_download
root = maybe_download('gs://openpi-assets/checkpoints/pi05_base', gs={'token': 'anon'})
assert (root / 'params' / '_METADATA').is_file(), root
print('Pretrained initialization:', root / 'params')
PY
export WORLDLA_INIT_PARAMS="$WORLDLA_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"

# Also works if you skipped the full fine-tuned model download in section 2.
huggingface-cli download "$WORLDLA_HF_REPO" --revision "$WORLDLA_HF_REVISION" \
  --include 'original/assets/paligemma_tokenizer.model' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_TOKENIZER="$WORLDLA_RELEASE_ROOT/original/assets/paligemma_tokenizer.model"
export WORLDLA_OUTPUT_ROOT="$WORLDLA_PROJECT_ROOT/outputs/reproduction_01"
```

Do **not** set `WORLDLA_INIT_PARAMS` to the fine-tuned `original/params` or
`merged/params`. The source run's initialization fingerprints are in
`dataset_alignment/pi05_base_files_sha256.json`. Check them before training:

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['WORLDLA_INIT_PARAMS'])
index = Path(os.environ['WORLDLA_REPRO_ROOT']) / 'dataset_alignment/pi05_base_files_sha256.json'
for name, expected in json.loads(index.read_text()).items():
    path = root / name
    assert path.stat().st_size == expected['bytes'], name
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == expected['sha256'], name
print('Pretrained initialization fingerprints OK')
PY
```

The released YAMLs consume **five absolute-path variables**:
`WORLDLA_REPRO_ROOT`, `WORLDLA_LEROBOT_ROOT`, `WORLDLA_INIT_PARAMS`,
`WORLDLA_TOKENIZER` and `WORLDLA_OUTPUT_ROOT`. Dataset ID and normalization
asset ID both resolve to `robotwin_official50_demo_clean_v21`.

## 7. Train Router, then action adaptation

### Check the pipeline first

After preparing all inputs, run four updates per stage on **four idle GPUs**:

```bash
python scripts/smoke_training.py \
  --lerobot-root "$WORLDLA_LEROBOT_ROOT" \
  --dataset-repo-id robotwin_official50_demo_clean_v21 \
  --asset-dir "$WORLDLA_REPRO_ROOT/assets" \
  --asset-id robotwin_official50_demo_clean_v21 \
  --route-manifest "$WORLDLA_REPRO_ROOT/stage_route_manifest.jsonl" \
  --init-params "$WORLDLA_INIT_PARAMS" --tokenizer-path "$WORLDLA_TOKENIZER" \
  --gpus 0,1,2,3 --batch-size 4 --steps 4 \
  --output "$WORLDLA_PROJECT_ROOT/outputs/train_smoke_01"
```

The helper refuses existing outputs or occupied GPUs. Check
`outputs/train_smoke_01/status.json`, `router.log` and `action.log`. It verifies
finite losses/nonzero gradients, LoRA/adapter updates, a frozen restored Router,
and checkpoint restoration, not full convergence or task SR.

### Main 8-GPU recipe

Use eight **idle** GPUs and a fresh output root. Keep `set -euo pipefail` active
so Router failure does not start action training. Run these foreground jobs
in a persistent terminal/session:

```bash
mkdir -p "$WORLDLA_OUTPUT_ROOT/logs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/router_60k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/router.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/action_100k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/action.log"
```

| Stage | Updates | Trainable components | Global batch / learning rate |
| --- | --- | --- | --- |
| Router | 60,000 | Router only; VLA backbone frozen | 32 / constant 2e-4, cached-route KL |
| Action | 100,000 | VLM/action-expert LoRA + message adapter/gates; Router frozen | 32 / warmup 3k, plateau to 50k, cosine decay to 2e-5 |

Both use seed 20260901 and FSDP=8. These are **main release** budgets, not
teacher-ablation 20k/40k budgets. Use the **downloaded YAMLs**, not generic
`configs/` templates with different example paths/IDs. CLI flags override YAML.
With fewer GPUs, override `--fsdp-devices` in **both** commands and use a
divisible global batch; that hardware setup is not a separately validated
full-budget reproduction.

Logs and checkpoints:

```text
outputs/reproduction_01/
  logs/{router,action}.log
  router_60k/
    experiment.json, train_metrics.jsonl, validation_metrics.jsonl
    router_checkpoints/final_step_60000/
  action_100k/
    experiment.json, train_metrics.jsonl, validation_metrics.jsonl
    action_checkpoints/final_step_100000/
    eval_snapshots/step_060000/
    eval_snapshots/step_080000/
```

```bash
tail -f "$WORLDLA_OUTPUT_ROOT/logs/action.log"
tail -n 5 "$WORLDLA_OUTPUT_ROOT/action_100k/validation_metrics.jsonl"
```

Wait for the Router's `experiment.json` to say `completed: true` before action
training. The action YAML refers to the Router **run directory**, not an NPZ
or raw checkpoint subfolder. Nonempty outputs are refused by default; do not
use `--overwrite` as a retry mechanism. Checkpoints are **parameter-only**:
`--resume-model-dir` warm-starts and resets optimizer moments; it is not
seamless full-state resume. Training/validation loss is not task SR.

## 8. Export your trained checkpoint

The action run contains adaptation parameters, not a self-contained deployment
folder. Export with its matching base, tokenizer and normalization assets.
Use new output directories; source checkpoints remain read-only:

```bash
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant original --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-original"

# Optional: a second, LoRA-folded full checkpoint.
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant merged --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-merged"

export WORLDLA_MODEL_DIR="$WORLDLA_PROJECT_ROOT/outputs/release-original"
CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" --dummy
```

Reuse **section 3** with this `WORLDLA_MODEL_DIR` to evaluate the new policy.
Exports include `params/`, `config.json`, `assets/`, `files_sha256.json` and
`export_report.json`; deployment needs no external base or Fast-WAM.
After moving a training run, use `--base-params`, `--action-params` and
`--assets-dir` to fix source locations, not silently substitute another base.
Export checks parameter round trips/source fingerprints, not rollout success.

## 9. Code tests and validation scope

In the training/development environment, from the repository root:

```bash
python -m pip install 'pytest>=8' 'build>=1'
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m pytest -q
python -m build --wheel --no-isolation
python scripts/check_wheel.py dist/worldla-0.2.0-py3-none-any.whl
```

The 0.2 namespace migration passed **32 CPU tests**, the 89-leaf parameter-layout
check, fixed-input regressions for both formats, four real-data training
updates per stage, and paired `press_stapler/demo_clean` smoke evaluation
(3/3 each). These are **not** new full-budget convergence or 50-task SR tests.
See [validation records](docs/VALIDATION.md) and [reproducibility boundaries](docs/REPRODUCIBILITY.md).

All live source uses the `worldla` namespace; no `openpi`/`openpi_client` Python
package is required. Parameter keys are preserved for checkpoint compatibility;
attribution and the [migration record](docs/NAMESPACE_MIGRATION.md) remain.
Raw Fast-WAM discovery/extraction instrumentation is not bundled: this release
supports **cached-target student retraining**, not regenerating teacher targets
from scratch. R2 pooled-feature ablations need additional genuine teacher
features, not this route bundle.

## License

Project code: Apache-2.0. See `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`.
Weights, tokenizer, datasets and simulator assets retain their upstream terms;
the code license does not relicense them.
