# WorldLA

WorldLA 将 Fast-WAM 的动作—视觉路由关系迁移到高效的 π0.5 策略，
并将新计划与时间对齐的旧计划尾部融合。Fast-WAM 只参与离线监督构建，
使用发布缓存训练和部署时均不需要运行教师模型。

[论文](docs/WorldLA.pdf) · [English](README.md) ·
[Hugging Face 权重与训练缓存](https://huggingface.co/xhx0327/WorldLA-100k-decay-full)

| 模型 | RoboTwin clean | RoboTwin randomized | 平均 |
|---|---:|---:|---:|
| WorldLA | 82.18% | 46.74% | 64.46% |

评测覆盖50项任务，每种配置下每项任务100回合。

**使用现有权重：第1–3节。重新训练：第1节及第4–7节。**

## 1. 安装

环境要求：Linux、Python 3.11、NVIDIA GPU、兼容 CUDA 12 的驱动。
以下命令使用 Bash，在仓库根目录执行。

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

Hugging Face 仓库公开，无需登录。使用依赖文件中的固定版本；
打开新终端时，重新设置上述路径变量。

## 2. 下载权重并推理

下载完整检查点，推理时无需另外准备基座模型。

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'original/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_MODEL_DIR="$WORLDLA_RELEASE_ROOT/original"
```

发布了两种格式：

| 目录 | 内容 |
|---|---|
| `original/` | 完整基座、独立 LoRA、路由器、消息适配器、门控和推理资源；推荐用于复现 |
| `merged/` | LoRA 融入主干权重，保留路由器和消息适配器 |

使用融合版时，下载 `merged/*` 并将 `WORLDLA_MODEL_DIR` 指向 `merged/`。
两种格式均由 WorldLA 的 Flax/Orbax 加载器读取。

输入 NPZ 包含 `state`（float32，`[14]`）及 `cam_high`、`cam_left_wrist`、
`cam_right_wrist`（RGB uint8，`[3,H,W]`）。状态与动作使用发布权重对应的
ALOHA 关节和夹爪布局。

```bash
CUDA_VISIBLE_DEVICES=0 worldla infer \
  --model-dir "$WORLDLA_MODEL_DIR" \
  --input /absolute/path/to/observation.npz \
  --prompt 'Place the can in the basket.' \
  --output "$WORLDLA_PROJECT_ROOT/outputs/inference/actions.npz"
```

输出为 `[50,14]` 动作块。闭环接入时，`WorldLAPolicy.plan(observation)`
返回融合后的十步动作；每回合开始调用 `reset()`，执行完十步后获取新观测，
再调用 `plan()`。

## 3. RoboTwin 评测

按照 [RoboTwin 官方安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
安装仿真环境、规划器及资源。WorldLA 与 RoboTwin 使用独立的 Python 环境。
当前接口对应以下版本：

- RoboTwin：`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`。
- XPolicyLab：`0ccd8e9f3ed76c1a3186a5b151bead5b53cf2249`。

### 终端 A：启动模型服务

激活 WorldLA 的 `.venv`，并设置第1–2节的路径。

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

保持服务运行。一个服务对应一个评测流；独立并行评测使用不同服务和端口。
远程访问使用 SSH 隧道，不要将未经认证的服务直接暴露到公网。

### 终端 B：运行100回合评测

在 WorldLA 仓库中激活 `.venv`，并指定 RoboTwin 环境的 Python：

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

其他任务修改 `--task`；随机化评测将 `--task-config` 改为 `demo_randomized`。
50项任务列表位于 `<RoboTwin>/env_cfg/eval/all_tasks.yml`。
Clean 使用 `seen` 指令，randomized 使用 `unseen` 指令。

每项任务的结果写入：

```text
<RoboTwin>/eval_result/<task>/WorldLA/<task_config>/<run_tag>/<timestamp>/_result.txt
```

默认预测50步，去噪10次，每次重新规划后执行10步。
机械臂旧/新计划权重为0.75/0.25，夹爪采用离散切换。
在服务端设置 `--old-plan-weight 0` 可关闭机械臂计划融合。

## 4. 下载训练缓存

重新训练需要三类输入：发布的路由缓存、与缓存匹配的机器人演示、
π0.5 预训练初始化权重。

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

包内提供20,000条派生帧锚点监督、按回合隔离的训练/验证/测试划分
（16,000/2,000/2,000条锚点）、归一化资源、转换工具和训练 YAML。
每个路由目标为 `[4,768]`：四个相对动作位置分组，对应三个视角的16×16视觉网格。
机器人图像和动作在第5节另行下载。

## 5. 准备并对齐机器人数据

使用官方50项任务、每项50条演示的 `demo_clean` 数据：
[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
中的 `dataset/<task>/demo_clean.zip`。
以下单独创建数据工具目录：

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

对齐检查核对缓存与数据的回合/帧顺序、指令和状态/动作指纹。
预期共2,500回合、548,893帧；动作训练使用训练回合的全部439,140帧。
对齐通过后再训练。数据转换中断时，可添加 `--resume` 继续。

## 6. 初始化并训练

从 JAX 格式的 π0.5 预训练基座开始训练，而不是使用已微调的 WorldLA 权重。

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

已有基座时，将 `WORLDLA_INIT_PARAMS` 指向其 `params/` 目录，并运行同一指纹检查。
每次训练使用新的 `WORLDLA_OUTPUT_ROOT`。发布配置读取 `WORLDLA_REPRO_ROOT`、
`WORLDLA_LEROBOT_ROOT`、`WORLDLA_INIT_PARAMS`、`WORLDLA_TOKENIZER`、
`WORLDLA_OUTPUT_ROOT` 五个路径变量。

在八张 GPU 上先训练 Router，再进行动作适配：

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

| 阶段 | 步数 | 更新参数 | Batch / 学习率 |
|---|---:|---|---|
| Router | 60,000 | Router；冻结 VLA 主干 | 32 / 固定2e-4 |
| 动作适配 | 100,000 | VLM/动作专家 LoRA、消息适配器和门控；冻结 Router | 32 / 预热3k，保持至50k，余弦衰减至2e-5 |

两阶段均使用 seed 20260901、FSDP=8。使用下载包中的数据集专用 YAML。
少于八卡时，两条命令都需设置 `--fsdp-devices`，且全局 batch 可被设备数整除。

日志位于 `$WORLDLA_OUTPUT_ROOT/logs/`。最终参数检查点为：

```text
router_60k/router_checkpoints/final_step_60000/
action_100k/action_checkpoints/final_step_100000/
```

动作配置从 `router_60k` 实验目录恢复 Router。
检查点保存模型参数；使用 `--resume-model-dir` 开始新训练时会重置优化器状态。

## 7. 导出并评测训练后的模型

```bash
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant original --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-original"
export WORLDLA_MODEL_DIR="$WORLDLA_PROJECT_ROOT/outputs/release-original"
```

导出包含基座、训练模块、tokenizer 和归一化资源。
需要融合版时，将 `--variant` 改为 `merged`，并使用另一输出目录。
随后按第3节评测新的 `WORLDLA_MODEL_DIR`。

## 许可证

代码采用 Apache-2.0，见 [LICENSE](LICENSE)、[NOTICE](NOTICE)、
[第三方声明](THIRD_PARTY_NOTICES.md)。权重、tokenizer、数据及仿真资源
遵循各自的原始许可。
