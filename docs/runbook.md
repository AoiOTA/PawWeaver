# 运行手册

从 `/home/lyb/pawweaver` 运行。若 shell 继承 ROS 的 `PYTHONPATH`，在 Python 命令前加 `env -u PYTHONPATH PYTHONNOUSERSITE=1`；测试脚本已处理这一点。

## 资产与预览

```bash
conda activate pawweaver-runtime
pawweaver fetch-assets
pawweaver build-assets --diagnostic
MUJOCO_GL=egl python scripts/render_asset.py assets/generated/diagnostic --video
conda activate pawweaver-train
python scripts/convert_usd.py assets/generated/diagnostic --diagnostic --output-subdir fixed_groups_20260909 --headless
python scripts/check_isaac_robot.py assets/generated/diagnostic --contacts --headless
```

运行器从 `usd/conversion.json` 读取实际 USD 入口及校验值，不依赖导入器生成的目录编号。`--output-subdir` 必须指定新的子目录（上例已存在，复现请换新名称）；转换、固定组关系验证和哈希全部成功后才原子替换该 locator，旧 USD 文件保持不变。转换器只为 canonical 固定连接分量内的碰撞 link 添加显式 pair filter，穿过无质量固定中间 frame，不跨可动关节、不合并 link；实际 link/prim 对及 producer 哈希记录在 conversion metadata。`--fixed-base` 单独输出 `usd-fixed/`，用于执行器响应检查。

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

## 真实几何的最小工程诊断

M0 未闭合时，只能显式提供带来源的临时执行器参数。复用已有组合几何/USD，不重建资产：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python scripts/train.py --asset assets/generated/diagnostic --diagnostic --provisional-spec configs/diagnostic_actuators.json --config configs/diagnostic_training.json --output artifacts/runs/diagnostic_geometry_fixed_groups_seed0 --seed 0 --num-envs 4 --iterations 1 --headless
```

该配置为 16 步、2 epoch、2 minibatch，关闭预测、速度估计、自适应采样、随机化和示范。参数文件绑定当前资产哈希，按关节名使用 canonical URDF 的位置/速度限位；增益/力矩为记录的官方候选参数，臂力矩仅是 CAN 编码范围，六个臂关节的 armature 现取 0.005 kg·m²，来自固定版本标准 Piper 厂商仿真并临时迁移至 Piper-H，不是 Piper-H 实测电机反射惯量；仅此参数替代零假设，臂 damping/frictionloss 仍为零且未经验证。腿部初始姿态仍取参考姿态；臂使用固定版本 SDK 的六零参考 `[0,0,0,0,0,0]`，控制器保持启用。来源为 `pyAgxArm@e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b` 的 Piper-H demo `test1.py` 第 104/127 行 `move_j([0.0]*robot.joint_nums)`；这不是已验证的 GUI/固件厂家 home 数值或断电姿态。动作尺度 0.1 rad、零延迟均为工程选择，不能声明硬件有效。

`initial_base_height_m` 为可选场景配置，缺省仍为 0.5 m。诊断配置取 0.317621507 m：由相同腿姿态下的实际 28 mm 足部碰撞球测得 0.183378493 m 悬空间隙，再保留 1 mm 初始间隙。重置复用该场景初始根位姿；Isaac 初始化不引入 MuJoCo 依赖。来源记录在诊断配置中。`render_asset.py` 读取同一诊断参数文件中的 q0；`artifacts/preview/robot.png` 已更新为六零参考。历史候选同角度图片和静态接触结果见 `artifacts/preview/initial_pose_candidates/`；静态无非相邻接触不能证明动态或硬件安全。

`run.json` 保存完整临时参数、来源、资产/USD 标识、初始足端高度和传感器映射，并按关节名记录从 PhysX 直接读取且核对的实际 armature；`metrics.jsonl` 保存有限性检查、每个物理子步的力矩饱和、TCP 误差、跌倒/重置、吞吐和优化器实测更新。检查点禁止混合诊断/正式模式或不同临时参数/来源。导出包始终 `trained=false`，正式评估会拒绝它。此次输出已存在，复现实验请使用新的输出目录以保留记录。

当前正确编译被动参数后的保持证据见 `artifacts/runs/diagnostic_mujoco_constants_revalidation/`：默认 `.1` 零动作 10 秒 MuJoCo 终态俯仰约 -13.355°，两个后大腿接地，不能称为稳定站立；精确重放 `.2` 腿动作尺度的常量动作候选终态约 +0.304°、全程仅四足接地，但 J2/J3 限位反力仍持续存在，不是主动臂策略或硬件验收。旧 -14.16° 等编译后改 armature 的 artifact 数值已被此重验替代。新 USD 已移除固定链内的臂法兰/相机支架虚假接触，不能把两引擎站姿差异归给该接触。相同默认输入在 0.02–2.00 秒的重叠记录仍有最大基座高度差 7.615 mm；真实被动响应等价性未闭合。候选仅在 artifacts 中，诊断默认仍为 `.1`。

可在新的 artifact 目录复用有界保持脚本（保持默认 PD/50 Hz/2 ms 不变）：

```bash
mkdir artifacts/runs/reproduce_constants_hold10s
cp artifacts/runs/diagnostic_mujoco_constants_revalidation/holds.py artifacts/runs/reproduce_constants_hold10s/holds.py
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python artifacts/runs/reproduce_constants_hold10s/holds.py
mkdir -p artifacts/runs/reproduce_physx_hold2s
cp artifacts/runs/diagnostic_physx_sdk_zero_armature_hold2s/probe.py artifacts/runs/reproduce_physx_hold2s/probe.py
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python artifacts/runs/reproduce_physx_hold2s/probe.py --headless
```

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
