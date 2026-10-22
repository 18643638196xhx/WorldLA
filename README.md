# WorldLA

WorldLA transfers Fast-WAM's action–visual routing relations to an efficient
π0.5 policy and combines new plans with time-aligned tails of the previous plan.
Fast-WAM is only used to construct offline supervision; it is not needed to train
from the released cache or to run the policy.

[Paper](docs/WorldLA.pdf) · [中文](README_zh.md) ·
[Weights and training cache](https://huggingface.co/xhx0327/WorldLA-100k-decay-full)

| Model | RoboTwin clean | RoboTwin randomized | Average |
|---|---:|---:|---:|
| WorldLA | 82.18% | 46.74% | 64.46% |

Our evaluation covers 50 tasks, with 100 trials per task in each configuration.

**Use pretrained weights:** sections 1–3. **Retrain WorldLA:** sections 1 and 4–7.

## 1. Installation

Requirements: Linux, Python 3.11, an NVIDIA GPU and a CUDA-12-compatible driver.
Run the commands below in Bash, from the repository root.

```bash
git clone https://github.com/18643638196xhx/WorldLA.git
cd WorldLA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps

export WORLDLA_PROJECT_ROOT="$(pwd)"
export WORLDLA_HF_REPO=xhx0327/WorldLA-100k-decay-full
export WORLDLA_HF_REVISION=7fd1111c6fa8419a64bb6f26383b5025ffbf436d
export WORLDLA_RELEASE_ROOT="$WORLDLA_PROJECT_ROOT/checkpoints/worldla-100k"
set -euo pipefail
```

The Hugging Face release is public. Keep the provided dependency versions.
Re-export the path variables when opening another terminal.

## 2. Download weights and run inference

Download the full checkpoint; no separate base model is needed for inference.

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'original/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_MODEL_DIR="$WORLDLA_RELEASE_ROOT/original"
```

The release provides two formats:

| Folder | Contents |
|---|---|
| `original/` | Full base weights, separate LoRA, router, message adapter, gates and inference assets; recommended for reproduction |
| `merged/` | Full model with LoRA folded into dense weights; router and message adapter retained |

For the merged format, download `merged/*` instead and point `WORLDLA_MODEL_DIR`
to the `merged/` folder. Both formats use WorldLA's Flax/Orbax loader.

Prepare an observation NPZ containing `state` (float32, `[14]`) and `cam_high`,
`cam_left_wrist`, `cam_right_wrist` (RGB uint8, `[3,H,W]`). State and actions use the
released ALOHA joint/gripper layout.

```bash
CUDA_VISIBLE_DEVICES=0 worldla infer \
  --model-dir "$WORLDLA_MODEL_DIR" \
  --input /absolute/path/to/observation.npz \
  --prompt 'Place the can in the basket.' \
  --output "$WORLDLA_PROJECT_ROOT/outputs/inference/actions.npz"
```

The output contains a `[50,14]` action chunk. For closed-loop integration,
`WorldLAPolicy.plan(observation)` returns the next ten commands with plan fusion;
call `reset()` at the start of each episode and acquire a new observation after
executing those ten commands.

## 3. Evaluate on RoboTwin

Install RoboTwin, its simulator environment, planners and assets using the
[official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
WorldLA and RoboTwin use separate Python environments. The integration targets:

- RoboTwin: `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
- XPolicyLab: `0ccd8e9f3ed76c1a3186a5b151bead5b53cf2249`.

### Terminal A: start the policy server

Activate WorldLA's `.venv` and set the paths from sections 1–2.

```bash
python -m pip install -r requirements-xpolicylab.txt
export WORLDLA_ROBOTWIN_ROOT=/absolute/path/to/installed/RoboTwin
export WORLDLA_XPOLICYLAB_ROOT="$WORLDLA_ROBOTWIN_ROOT/XPolicyLab"
mkdir -p outputs/eval

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  worldla serve-xpolicylab \
  --xpolicylab-root "$WORLDLA_XPOLICYLAB_ROOT" \
  --model-dir "$WORLDLA_MODEL_DIR" --host 127.0.0.1 --port 19000 \
  --inference-seed 20260901 --old-plan-weight 0.75 \
  2>&1 | tee outputs/eval/policy.log
```

Keep the server running. It serves one evaluation stream; independent parallel
streams use separate servers and ports. Bind remote deployments through an SSH
tunnel rather than exposing the unauthenticated server publicly.

### Terminal B: run 100 evaluation episodes

In the WorldLA repository, activate `.venv` and specify RoboTwin's Python:

```bash
source .venv/bin/activate
export WORLDLA_ROBOTWIN_ROOT=/absolute/path/to/installed/RoboTwin
export WORLDLA_ENV_PYTHON=/absolute/path/to/robotwin-env/bin/python

CUDA_VISIBLE_DEVICES=0 worldla eval-xpolicylab \
  --robotwin-root "$WORLDLA_ROBOTWIN_ROOT" \
  --env-python "$WORLDLA_ENV_PYTHON" \
  --host 127.0.0.1 --port 19000 --task press_stapler \
  --task-config demo_clean --episodes 100 --seed-group 0 \
  --run-tag worldla_100k_decay \
  2>&1 | tee outputs/eval/press_stapler_clean.log
```

Change `--task` for another task and `--task-config` to `demo_randomized` for randomized
evaluation. The 50-task list is `<RoboTwin>/env_cfg/eval/all_tasks.yml`.
Clean uses `seen` instructions; randomized uses `unseen` instructions.

RoboTwin writes each result to:

```text
<RoboTwin>/eval_result/<task>/WorldLA/<task_config>/<run_tag>/<timestamp>/_result.txt
```

Deployment uses a 50-step horizon, ten denoising iterations and ten executed
actions per replan. Arm fusion uses old/new weights 0.75/0.25; grippers switch
discretely. Set `--old-plan-weight 0` on the server to disable arm-plan fusion.

## 4. Download the training cache

Retraining requires three inputs: the released route cache, its matching
robot demonstrations, and pretrained π0.5 initialization.

```bash
python -m pip install -r requirements-train.txt h5py==3.12.1
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" \
  --include 'training/stage_routes_20k/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"

export WORLDLA_ROUTE_ARCHIVE="$WORLDLA_RELEASE_ROOT/training/stage_routes_20k/stage_routes_20k.tar.gz"
printf '%s  %s\n' \
  43ab0e7cb9d2fec12813a351bedb9957af1847cb2a1601c68af5e1f9a5a412dc \
  "$WORLDLA_ROUTE_ARCHIVE" | sha256sum -c -
mkdir -p "$WORLDLA_PROJECT_ROOT/data"
test ! -e "$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
tar -xzf "$WORLDLA_ROUTE_ARCHIVE" -C "$WORLDLA_PROJECT_ROOT/data"
export WORLDLA_REPRO_ROOT="$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
JAX_PLATFORMS=cpu python "$WORLDLA_REPRO_ROOT/tools/verify_bundle.py" \
  --root "$WORLDLA_REPRO_ROOT" --worldla-loader
```

The bundle contains 20,000 derived frame-anchor targets, episode-disjoint
train/validation/test splits (16,000/2,000/2,000 anchors), normalization assets,
conversion tools and the training YAMLs. Each route target has shape `[4,768]`:
four relative action-position bins over three 16×16 visual-token grids.
Robot images/actions are downloaded separately in section 5.

## 5. Prepare and align the robot dataset

Use the official 50-task `demo_clean` dataset, 50 demonstrations per task:
`dataset/<task>/demo_clean.zip` from
[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).
The following uses a separate checkout for data tools:

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

The alignment check matches cached targets to episode/frame order, prompts and
state/action fingerprints. The expected dataset has 2,500 episodes and 548,893
frames; action training uses all 439,140 frames in the training episodes.
Proceed after alignment passes. Add `--resume` to the converter after an interruption.

## 6. Initialize and train

Training starts from the JAX π0.5 base, rather than the fine-tuned WorldLA export.

```bash
export WORLDLA_DATA_HOME="$WORLDLA_PROJECT_ROOT/data/pretrained_cache"
python - <<'PY'
from worldla.utils.downloads import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi05_base', gs={'token': 'anon'})
PY
export WORLDLA_INIT_PARAMS="$WORLDLA_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"

huggingface-cli download "$WORLDLA_HF_REPO" --revision "$WORLDLA_HF_REVISION" \
  --include 'original/assets/paligemma_tokenizer.model' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_TOKENIZER="$WORLDLA_RELEASE_ROOT/original/assets/paligemma_tokenizer.model"
export WORLDLA_OUTPUT_ROOT="$WORLDLA_PROJECT_ROOT/outputs/reproduction_01"

python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['WORLDLA_INIT_PARAMS'])
index = Path(os.environ['WORLDLA_REPRO_ROOT']) / 'dataset_alignment/pi05_base_files_sha256.json'
for name, expected in json.loads(index.read_text()).items():
    path = root / name
    assert path.stat().st_size == expected['bytes'], name
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name
print('Pretrained initialization verified')
PY
```

For an existing base download, set `WORLDLA_INIT_PARAMS` to its `params/` directory
and run the same fingerprint check. Use a new `WORLDLA_OUTPUT_ROOT` for each run.
The released YAMLs use `WORLDLA_REPRO_ROOT`, `WORLDLA_LEROBOT_ROOT`,
`WORLDLA_INIT_PARAMS`, `WORLDLA_TOKENIZER` and `WORLDLA_OUTPUT_ROOT`.

Run Router training first, then action adaptation on eight GPUs:

```bash
set -euo pipefail
mkdir -p "$WORLDLA_OUTPUT_ROOT/logs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/router_60k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/router.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/action_100k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/action.log"
```

| Stage | Steps | Updated parameters | Batch / learning rate |
|---|---:|---|---|
| Router | 60,000 | Router; VLA backbone frozen | 32 / constant 2e-4 |
| Action adaptation | 100,000 | VLM/action-expert LoRA and message adapter/gates; Router frozen | 32 / 3k warmup, constant to 50k, cosine decay to 2e-5 |

Both configurations use seed 20260901 and FSDP=8. Use the dataset-specific YAMLs
from the downloaded bundle. With fewer GPUs, set `--fsdp-devices` in both commands
and keep the global batch divisible by the device count.

Logs are in `$WORLDLA_OUTPUT_ROOT/logs/`. Final parameter checkpoints are:

```text
router_60k/router_checkpoints/final_step_60000/
action_100k/action_checkpoints/final_step_100000/
```

The action YAML restores the router from the `router_60k` run directory.
Checkpoints contain parameters; `--resume-model-dir` resets optimizer state when
warm-starting a new run.

## 7. Export and evaluate your trained model

```bash
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant original --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-original"
export WORLDLA_MODEL_DIR="$WORLDLA_PROJECT_ROOT/outputs/release-original"
```

The export includes base weights, trained modules, tokenizer and normalization
assets. Use `--variant merged` and another output directory for a LoRA-folded
export. Evaluate the exported `WORLDLA_MODEL_DIR` using section 3.

## License

Code: Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights, tokenizer, datasets
and simulator assets retain their respective upstream terms.
