# 运行手册

从 `/home/lyb/pawweaver` 运行。若 shell 继承 ROS 的 `PYTHONPATH`，在 Python 命令前加 `env -u PYTHONPATH PYTHONNOUSERSITE=1`；测试脚本已处理这一点。

## 资产与预览

```bash
conda activate pawweaver-runtime
pawweaver fetch-assets
pawweaver build-assets --diagnostic
MUJOCO_GL=egl python scripts/render_asset.py assets/generated/diagnostic --video
conda activate pawweaver-train
python scripts/convert_usd.py assets/generated/diagnostic --diagnostic --headless
python scripts/check_isaac_robot.py assets/generated/diagnostic --contacts --headless
```

运行器从 `usd/conversion.json` 读取实际 USD 入口及校验值，不依赖导入器生成的目录编号。`--fixed-base` 单独输出 `usd-fixed/`，用于执行器响应检查。

## 软件集成测试

```bash
conda activate pawweaver-runtime
python scripts/make_test_fixture.py
conda activate pawweaver-train
python scripts/convert_usd.py artifacts/software-fixture --headless
python scripts/convert_usd.py artifacts/software-fixture --fixed-base --headless
python scripts/check_training_pipeline.py --randomize --headless
python scripts/check_pd_response.py artifacts/software-fixture --output artifacts/software-fixture/pd --headless
bash scripts/check.sh
conda activate pawweaver-runtime
python scripts/check_runtime_pipeline.py
MUJOCO_GL=egl python scripts/check_rendered_vision.py
MUJOCO_GL=egl python scripts/check_visual_pipeline.py
MUJOCO_GL=egl bash scripts/check.sh
```

合成机器人不含 AS2/Piper 几何，策略包 `trained=false`，不能用于正式验收。视觉录屏是极短的软件测试，不能作为跟踪性能演示。

## 正式训练与课程

以下命令要求 M0 参数闭合，当前不能成功开始正式训练：

```bash
conda activate pawweaver-runtime
pawweaver build-assets --output assets/generated/verified
conda activate pawweaver-train
python scripts/convert_usd.py assets/generated/verified --headless
python scripts/train.py --asset assets/generated/verified --output artifacts/runs/baseline_seed0 --seed 0 --num-envs 1024 --iterations 1000 --headless
```

`--resume checkpoint.pt` 恢复优化器、采样器与随机数状态，但重新开始仿真回合，不声称完全逐步续跑；配置及资产必须一致。`--initialize-from checkpoint.pt` 只读取兼容 Actor/Critic 参数，供下一课程阶段使用，重新创建优化器。

`scripts/prepare_experiments.py` 生成三个种子、预测/自适应采样开关、四阶段课程的配置与命令，默认 48 个作业，不启动训练。提供训练集 `--demonstrations` 后增加示范数据消融，共 96 个作业。是否晋级由固定验证结果决定，不能仅看训练奖励或迭代数。

## 固定测试集与双引擎评估

```bash
conda activate pawweaver-data
python -m pawweaver.evaluation create artifacts/evaluation/suite
conda activate pawweaver-train
python scripts/evaluate_isaac.py --asset assets/generated/verified --bundle artifacts/runs/baseline_seed0/bundle --suite artifacts/evaluation/suite --seed 0 --output artifacts/evaluation/physx_seed0 --headless
conda activate pawweaver-runtime
python scripts/evaluate_mujoco.py --asset assets/generated/verified --bundle artifacts/runs/baseline_seed0/bundle --suite artifacts/evaluation/suite --seed 0 --output artifacts/evaluation/mujoco_seed0
python -m pawweaver.evaluation compare artifacts/evaluation/physx_seed0/report.json artifacts/evaluation/mujoco_seed0/report.json --output artifacts/evaluation/comparison_seed0.json
```

默认 100 回合，含 0.3–2 m 静态目标与 60 秒动态轨迹。候选集固定随机种子和文件哈希，尚未完成精确硬件可达性审核，因此不能直接给出验收结论。每个训练种子都须使用同一策略在两个引擎评估。

## FastUMI 与视觉

在 `pawweaver-data` 中运行 `python -m pawweaver.data --help`。输入需要 pose、逐帧时间戳、稳定 source ID，以及单位、sensor→TCP、source→task、速度/加速度、工作区边界的配置。转换器仅读取位姿，不下载视频，增强前按 source ID 划分数据。当前没有下载实测 FastUMI 示范集。

在 `pawweaver-runtime` 中运行 `python -m pawweaver.visual_runtime --help`。需要有效资产、策略包、轨迹与 scenario JSON。scenario 定义 `marker_id`、`marker_size_m`、`marker_to_goal`、`marker_rpy_rad`，可加入延迟、遮挡、深度缺失、位置噪声和随机种子；`--record` 输出腕部视频。

相机在 2 ms 物理时钟上调度约 30 Hz 图像，50 Hz 控制器消费到达的测量。真值仅用于场景生成和评分。持续失跟后暂停参考推进、保持有界关节目标；这种保持的整机稳定性仍需训练策略验证。基座定位来自仿真状态，不包含真机自主定位。
