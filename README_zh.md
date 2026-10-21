# WorldLA 独立代码项目

本项目把 WorldLA 的模型、缓存教师监督训练、推理和重规划从历史实验目录中整理出来。
不依赖旧服务器的绝对路径，不修改原训练工程，也不会自动启动训练、评测或上传。

0.2版命名迁移后已通过 32 项 CPU 测试，重命名前后的89项参数路径及形状一致。
两种格式的权重在四个真实观测上的推理结果，分别与其改名前参考输出完全一致。
未融合版和融合版均已完成 `press_stapler/demo_clean` 的真实闭环短评测，同样三个
种子均为 3/3 成功。这不是重新训练或完整50任务评测，详见 [验证范围](docs/VALIDATION.md)。
此外，独立安装包已通过真实数据上的四卡训练短测：Router 和动作阶段各4步，
包含非零梯度、参数更新、Router 冻结以及检查点保存/恢复验证；损失和验证指标也与
改名前的短测完全一致，尚未重新验证完整预算收敛。

## 项目结构

- `src/worldla/models/worldla.py`：`WorldLAModel`、阶段路由器、视觉消息与门控。
- `src/worldla/models/backbone.py`：`WorldLABackbone`，保留现有 π0.5 参数结构。
- `src/worldla/models/configuration.py`：`WorldLAConfig`。
- `src/worldla/policy.py`：完整检查点加载、输入输出变换和推理接口。
- `src/worldla/replanning.py`：旧计划尾部与新计划头部的时间对齐融合。
- `src/worldla/training/runner.py`：两阶段训练；先训练路由器，再冻结路由器训练动作侧。
- `src/worldla/data/`：阶段路由缓存、教师空间映射和轨迹锚点构建。
- `src/worldla/integrations/`：RoboTwin 策略适配器与外部评测器启动入口。
- `src/worldla/utils/`、`src/worldla/client/`：工具、检查点支持和客户端。
- `configs/`：路由器 60k、动作模型 decay 100k 及部署示例配置。
- `tests/`：接口、路由、重规划、配置和可移植性测试。

## 使用

Linux、Python 3.11；请创建新环境，不要改动正在运行任务的环境。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
worldla --help
```

现有完整权重与源代码分开存储。代码支持同一模型仓库中的 `original/` 和 `merged/`；
推荐用未融合版复现。输入是三路 CHW、RGB、uint8 图像，14 维关节状态和语言指令；
输出是关节动作，而不是末端位姿。`infer()` 输出 50 步新动作块，`plan()` 输出融合后
待执行的 10 步动作。每个回合调用 `reset()`，执行完这 10 步再重新规划。

详细下载、训练和导出命令见英文 [README](README.md)。
训练路径通过 YAML 或命令行指定；动作训练示例为 warmup 3k、保持到 50k，
随后余弦下降至 100k 的 2e-5，峰值学习率 2e-4。

## 重要边界

1. 训练里的四个阶段是动作块相对位置分组，不等同于自动识别的接近、抓取、搬运、放置。
2. Fast-WAM 仅用于离线教师缓存；缓存训练和部署均不运行教师。
3. 第一版已整理缓存消费和转换，但原始 Fast-WAM shared-K 发现/采集工具尚未完整迁入。
4. RoboTwin 仿真器及其资源需单独安装；评测入口不能代替实际场景验证。
5. 参数 warm-start 会重置优化器，不宣称支持完整优化器状态续训。
6. 这次整理本身不等于重新完成训练、打榜或模型成功率验证。

主项目代码采用 Apache-2.0。基座、tokenizer、教师权重、数据和仿真资源的许可需分别核对。

0.2版已统一为 `worldla` 模块、文件和类命名，不再提供或依赖 `openpi`、
`openpi_client` Python 包。原始版权、来源和修改映射仍保留，详见
[命名迁移说明](docs/NAMESPACE_MIGRATION.md)。现有参数键和权重格式不改名。
