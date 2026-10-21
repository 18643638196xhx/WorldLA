# WorldLA：测试与训练完整流程

[English](README.md) · [Hugging Face 权重与数据](https://huggingface.co/xhx0327/WorldLA-100k-decay-full) ·
[训练细节](docs/TRAINING.md) · [评测接口](docs/EVALUATION.md) · [验证记录](docs/VALIDATION.md)

WorldLA 将 Fast-WAM 的动作—视觉关系迁移到 π0.5：路由器学习不同动作阶段
**从哪里读取信息**，视觉内容仍来自学生自身；执行时对齐并融合旧计划尾部和新计划头部。
使用已发布缓存训练及部署时均不运行 Fast-WAM。

| 目标 | 需要准备 | 阅读顺序 |
| --- | --- | --- |
| 直接使用现有模型 | 代码 + 一份完整权重 | 第 1、2 节 |
| RoboTwin 闭环评测 | 完整权重 + 兼容的仿真器、评测器及资源 | 第 1–3 节 |
| 重新训练 | 路由缓存 + 对齐的机器人数据 + π0.5 预训练基座 | 第 1、4–7 节 |
| 导出新模型并评测 | 完成的动作训练目录及其初始化、归一化资源 | 第 8 节，再回到第 3 节 |

## 1. 安装独立代码

使用 **Linux、Python 3.11、兼容 CUDA 12 的 NVIDIA 驱动**。
创建新环境，不要升级其他正在训练或评测的环境。命令使用 **Bash**，
始终在 WorldLA 仓库根目录执行；路径变量均为绝对路径，新终端需要重新设置。

```bash
git clone https://github.com/18643638196xhx/WorldLA.git
cd WorldLA
export WORLDLA_PROJECT_ROOT="$(pwd)"
python3.11 -m venv .venv
. .venv/bin/activate
set -euo pipefail

# 只推理或评测：
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
worldla --help
```

需要**训练**时，可直接将上面的依赖文件换为 `requirements-train.txt`，
它已经包含 runtime 依赖；若已安装 runtime，则补充：

```bash
python -m pip install -r requirements-train.txt
python -m pip install h5py
```

依赖固定了 JAX 0.5.0、Flax 0.10.2、NumPy 1.26.4，以及 LeRobot 提交
`a445d9c9da6bea99a8972daa4fe1fdd053d711d2`，不要随意换成最新版 LeRobot。
仿真器依赖另行安装。

### 设置 Hugging Face 下载位置

仓库现已**公开**，无需登录。固定以下提交，避免混用不同版本的权重与训练材料：

```bash
export WORLDLA_HF_REPO=xhx0327/WorldLA-100k-decay-full
export WORLDLA_HF_REVISION=7fd1111c6fa8419a64bb6f26383b5025ffbf436d
export WORLDLA_RELEASE_ROOT="$WORLDLA_PROJECT_ROOT/checkpoints/worldla-100k"
```

材料记录的源代码提交是 `31301d2c33d90b86afab0e0a5168b780d24378c6`，
详见包内 `provenance.json`。只有下载环境要求认证时才执行 `huggingface-cli login`；
不要在命令、文档或 issue 中粘贴 token。

## 2. 下载完整权重并测试推理

| Hub 中的目录 | 大小 | 内容与用途 |
| --- | --- | --- |
| `original/` | 约 5 GiB | 完整基座、独立 LoRA、Router、Adapter/门控、tokenizer 与归一化；推荐用于复现 |
| `merged/` | 约 9.5 GiB | LoRA 融入 FP32 主干；Router/Adapter 仍独立存在 |
| `training/stage_routes_20k/` | 压缩包约 96.3 MiB | 路由监督、配置、归一化、数据对齐记录及工具；不含机器人图像和动作 |

推理只需要前两种中的**一种**，无需额外下载 π0.5 基座或 Fast-WAM：

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'original/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_MODEL_DIR="$WORLDLA_RELEASE_ROOT/original"

CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" --dummy
```

`--dummy` 只验证加载和输出形状，不启动机器人或仿真；**不能执行其输出**。
成功应输出有限值动作块 `[50,14]`。第一次包含加载/JIT 编译，不代表稳定推理延迟。
使用融合版时，将下载匹配改成 `'merged/*'`，模型目录改为
`"$WORLDLA_RELEASE_ROOT/merged"`。均为自定义 Flax/Orbax 格式，不是 Transformers
`AutoModel`；由安装的 WorldLA 包加载，不运行权重包中的旧 Python 实现。
融合会改变低精度计算，不能断言两种格式的 rollout 成功率完全相同。

真实输入 NPZ：`state` 为 float32 `[14]`；
`cam_high`、`cam_left_wrist`、`cam_right_wrist` 为 RGB uint8 `[3,H,W]`。
状态和动作使用对应的 ALOHA 关节/夹爪坐标，不是笛卡尔末端位姿。

```bash
CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" \
  --input /absolute/path/to/observation.npz \
  --prompt 'Place the can in the basket.' \
  --output "$WORLDLA_PROJECT_ROOT/outputs/inference/actions.npz"
```

输出文件已存在时拒绝覆盖。Python 闭环接口如下，运行前选择空闲 GPU：

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
policy.reset()  # 每回合重置采样状态，并清空旧计划。
commands = policy.plan(observation)  # [10,14]，融合后待执行的十步动作。
# 控制循环：执行完十步，获取新的 observation，再调用 plan()。
# 本例不会实际执行 commands。
```

`infer(observation)['actions']` 返回未经旧计划融合的 50 步新动作块；
`plan()` 返回融合后的 10 步。默认去噪 10 次，每次执行 10 步，
机械臂旧/新计划权重为 0.75/0.25；夹爪采用离散切换。

## 3. RoboTwin 任务评测

按照 [RoboTwin 上游说明](https://github.com/RoboTwin-Platform/RoboTwin)
单独安装仿真器、SAPIEN/Vulkan、机械臂/规划器依赖和**完整资源**。
评测 Python 环境必须同时能导入 WorldLA 与仿真器。

**当前兼容范围：**提供旧版 `get_model / eval / reset_model` 策略接口，
通过兼容的 `script/eval_policy.py` 启动；并未适配所有最新版 XPolicyLab 接口。
直接克隆最新版 RoboTwin 不代表其评测器必然兼容。接口不同需另行适配，
详见 [评测指南](docs/EVALUATION.md)，不能把 dry-run 当作已跑通仿真。

先指定**已安装且接口兼容的**评测工程，检查启动命令：

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

确认后删去 `--dry-run`，先跑三回合短测。正式单任务改成 `--episodes 100`，
同时使用新的 `--run-tag` 和 summary 路径；其他任务修改 `--task`。
随机化评测改为 `--task-config demo_randomized`，会使用 unseen 指令，
clean 则使用 seen 指令。二者分别统计，三回合短测不用于报告正式成功率。

配对复现还应传入 `--accepted-seed-dir /absolute/path/to/matching/accepted_seeds`。
这是仿真种子/指令缓存，**不是**训练路由缓存，也不包含在 20K 包中。
没有对应缓存时不能声称复现历史同种子的对比结果。固定仿真器、资源、指令、
种子和控制设置；不能静默跳过困难种子、删除杂物，也不能把初始化崩溃算成策略失败。
`--old-plan-weight 0` 关闭旧计划融合。

命令末尾添加 `2>&1 | tee outputs/eval/press_stapler.log` 保存日志；
新终端执行 `tail -f outputs/eval/press_stapler.log` 查看。
检查日志中的物理 PCI 渲染设备，CUDA 和 Vulkan 设备选择并非同一件事。
渲染器报错不代表评测完成。需要同 accepted seeds 检查两种权重时，
可使用 `scripts/smoke_evaluation.py`，完整参数见评测指南。

## 4. 下载并校验训练路由缓存

只需推理/评测可跳过第 4–7 节。重新训练必须同时具备：
**路由缓存、匹配的机器人原始数据、π0.5 预训练初始化**，不是只下载现成模型。

```bash
huggingface-cli download "$WORLDLA_HF_REPO" \
  --revision "$WORLDLA_HF_REVISION" --include 'training/stage_routes_20k/*' \
  --local-dir "$WORLDLA_RELEASE_ROOT"

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
# 使用新目录，不覆盖已有或修改过的数据包。
test ! -e "$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
tar -xzf "$WORLDLA_RELEASE_ROOT/training/stage_routes_20k/stage_routes_20k.tar.gz" \
  -C "$WORLDLA_PROJECT_ROOT/data"
export WORLDLA_REPRO_ROOT="$WORLDLA_PROJECT_ROOT/data/stage_routes_20k"
JAX_PLATFORMS=cpu python "$WORLDLA_REPRO_ROOT/tools/verify_bundle.py" \
  --root "$WORLDLA_REPRO_ROOT" --worldla-loader
```

包内提供 `samples/`、`stage_route_manifest.jsonl`、仅由训练集计算的
`train_stage_priors.npz`、归一化 `assets/`、数据对齐记录、训练 `configs/`、
转换/校验 `tools/`、校验和及来源记录。

预期结果：20,000 条锚点，2,500 个原始回合，train/val/test 为
**16,000/2,000/2,000**，按回合隔离；每个路由目标为 **[4,768]**。
20K 是派生帧锚点，不是新增 20K 条机器人演示。
动作阶段使用训练回合的全部 **439,140 帧**，不局限于 16K 路由训练锚点。
四个阶段是动作块相对位置分组，不等同于自动标注的接近、抓取、搬运、放置。

## 5. 下载机器人数据并恢复严格对齐的 LeRobot 数据集

图像和动作需要从 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
另行获取。使用每项 50 条演示的 **`dataset/<task>/demo_clean.zip`**，
不是 randomized-500 数据。使用单独的数据准备 checkout，避免改动第 3 节评测环境：

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

原始布局为 `<WORLDLA_RAW_ROOT>/demo_clean/<task>/aloha_agilex/{data,instruction}/`。
`WORLDLA_LEROBOT_ROOT` 是数据集目录的**上一级**。保持固定任务、回合、帧顺序、
指令选择、官方 RGB 解码器和关节坐标；不要交换图像通道，也不要替换成另一种
LeRobot 导出。转换中断可添加 `--resume`，不能通过覆盖数据集来跳过对齐检查。

校验器核对 2,500 回合的指令、帧顺序及 state/action 数值指纹，
预期共 **548,893 帧**；并未校验全部图像像素。
历史原始数据的 Hub 提交号未记录，因此对齐校验是必要步骤；失败后应停止排查，
不能继续用错位监督训练。磁盘还需容纳 ZIP、解压数据、LeRobot 数据和运行缓存，
不是只需要 96 MiB。

## 6. 下载预训练基座并设置训练路径

初始化来自 [Physical Intelligence 发布的 JAX π0.5](https://github.com/Physical-Intelligence/openpi)：
`gs://openpi-assets/checkpoints/pi05_base`，不能随意换成转换后的 PyTorch 权重。
已有下载时跳过下载命令，将 `WORLDLA_INIT_PARAMS` 指向实际的 `params/`。

```bash
export WORLDLA_DATA_HOME="$WORLDLA_PROJECT_ROOT/data/pretrained_cache"
python - <<'PY'
from worldla.utils.downloads import maybe_download
root = maybe_download('gs://openpi-assets/checkpoints/pi05_base', gs={'token': 'anon'})
assert (root / 'params' / '_METADATA').is_file(), root
print('Pretrained initialization:', root / 'params')
PY
export WORLDLA_INIT_PARAMS="$WORLDLA_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"

# 即使第 2 节未下载完整微调模型，也可单独取得 tokenizer。
huggingface-cli download "$WORLDLA_HF_REPO" --revision "$WORLDLA_HF_REVISION" \
  --include 'original/assets/paligemma_tokenizer.model' \
  --local-dir "$WORLDLA_RELEASE_ROOT"
export WORLDLA_TOKENIZER="$WORLDLA_RELEASE_ROOT/original/assets/paligemma_tokenizer.model"
export WORLDLA_OUTPUT_ROOT="$WORLDLA_PROJECT_ROOT/outputs/reproduction_01"
```

不要把已微调的 `original/params` 或 `merged/params` 当作初始基座。
训练前按发布的初始化文件指纹验证，读取大权重需要一定时间：

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

发布的 YAML 使用五个绝对路径变量：`WORLDLA_REPRO_ROOT`、`WORLDLA_LEROBOT_ROOT`、
`WORLDLA_INIT_PARAMS`、`WORLDLA_TOKENIZER`、`WORLDLA_OUTPUT_ROOT`。
数据集与归一化资源的 ID 均为 `robotwin_official50_demo_clean_v21`。

## 7. 先训练 Router，再训练动作侧

### 先做真实数据训练短测

准备完成后，在**四张空闲卡**上各阶段训练 4 步，检查数据与保存流程：

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

脚本拒绝已有输出，并检查 GPU 未被占用。查看
`outputs/train_smoke_01/status.json`、`router.log`、`action.log`。
它验证有限损失/非零梯度、LoRA/Adapter 更新、Router 恢复并冻结以及检查点恢复，
不是完整训练收敛或成功率验证。

### 正式八卡训练

使用八张空闲卡、新输出目录，保持 `set -euo pipefail` 生效，防止 Router
失败后仍继续训练动作侧。下面是前台命令，建议在持久终端会话中运行：

```bash
mkdir -p "$WORLDLA_OUTPUT_ROOT/logs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/router_60k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/router.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 worldla train \
  --config "$WORLDLA_REPRO_ROOT/configs/action_100k.yaml" \
  2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/action.log"
```

| 阶段 | 步数 | 更新部分 | 全局 batch / 学习率 |
| --- | --- | --- | --- |
| Router | 60,000 | 仅 Router，冻结 VLA 主干 | 32 / 固定 2e-4，缓存路由 KL |
| 动作适配 | 100,000 | VLM/动作专家 LoRA + 消息 Adapter/门控，冻结 Router | 32 / warmup 3k，保持至 50k，再余弦下降至 2e-5 |

两阶段均为 seed 20260901、FSDP=8；这是主模型预算，不是消融的 20k/40k。
使用**下载包中的 YAML**，不是仓库 `configs/` 中路径/数据集 ID 不同的通用示例。
命令行可覆盖 YAML。少于八卡时，两阶段都需修改 `--fsdp-devices`，并使用可整除的
全局 batch；这属于不同硬件配置，不能声称已经独立验证了相同预算下的完整复现。

输出结构：

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

Router 的 `experiment.json` 必须为 `completed: true` 后才能进入动作阶段。
动作 YAML 引用的是 Router **实验根目录**，不是 NPZ 或参数子目录。
默认拒绝非空输出，不要通过 `--overwrite` 重试。检查点仅保存参数；
`--resume-model-dir` 是重置优化器的 warm-start，不是包含优化器状态的无缝续训。
训练/验证 loss 不等于任务成功率。

## 8. 导出自己训练的完整模型并评测

动作训练目录保存适配参数，部署前需与匹配的预训练基座、tokenizer 和归一化
组装。输出使用新目录，原始检查点保持只读：

```bash
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant original --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-original"

# 可选：额外导出 LoRA 融合版本。
worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
  --variant merged --tokenizer "$WORLDLA_TOKENIZER" \
  --output-dir "$WORLDLA_PROJECT_ROOT/outputs/release-merged"

export WORLDLA_MODEL_DIR="$WORLDLA_PROJECT_ROOT/outputs/release-original"
CUDA_VISIBLE_DEVICES=0 worldla infer --model-dir "$WORLDLA_MODEL_DIR" --dummy
```

随后复用**第 3 节**的命令，即可评测新的 `WORLDLA_MODEL_DIR`。
导出包含 `params/`、`config.json`、`assets/`、`files_sha256.json` 和
`export_report.json`，部署无需额外基座或教师。
搬迁训练目录后可用 `--base-params`、`--action-params`、`--assets-dir` 修正源路径，
不能偷偷替换基座。导出校验参数往返一致性和源文件未改变，但不自动验证成功率。

## 9. 代码测试与已验证范围

在训练/开发环境、仓库根目录执行：

```bash
python -m pip install 'pytest>=8' 'build>=1'
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m pytest -q
python -m build --wheel --no-isolation
python scripts/check_wheel.py dist/worldla-0.2.0-py3-none-any.whl
```

0.2 命名迁移已验证：32 项 CPU 测试、89 项参数路径/形状、两种权重各自的固定输入
回归、真实数据各阶段 4 步训练，以及 `press_stapler/demo_clean` 同三种子短测
（两种权重各 3/3）。不等于重新完成完整预算训练或 50 项成功率验证，详见
[验证记录](docs/VALIDATION.md) 和 [复现边界](docs/REPRODUCIBILITY.md)。

运行源码均使用 `worldla` 命名，不需要 `openpi`/`openpi_client` Python 包。
参数键不改名以兼容现有权重，来源与 [迁移记录](docs/NAMESPACE_MIGRATION.md) 保留。
原始 Fast-WAM shared-K 发现/采集栈尚未整体纳入；此流程支持
**从缓存教师目标重新训练学生**，不是从零重新生成教师缓存。
R2 整体特征消融还需真实教师隐特征，不能用空间路由包替代。

## 许可

主代码采用 Apache-2.0，见 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md`。
基座、tokenizer、数据和仿真资源适用上游各自许可，代码许可证不替它们重新授权。
