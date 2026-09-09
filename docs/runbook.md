# 运行手册

从 `/home/lyb/pawweaver` 运行。若 shell 继承 ROS 的 `PYTHONPATH`，在 Python 命令前加 `env -u PYTHONPATH PYTHONNOUSERSITE=1`；测试脚本已处理这一点。

## 当前目标与运行边界

单个强化学习 Actor **仅跟踪世界系末端位姿轨迹（位置＋朝向）**，统一输出 12 腿＋6 臂关节位置目标，自主决定全身运动。无外部底盘速度命令；参考 MLM 的因果目标历史与可选预测，后续面向手持 UMI。基座速度仅作估计量或 Critic／训练标签。276 维位姿观测、schema-2 轨迹／bundle、Isaac/MuJoCo 和视觉 IO 已实现并完成 CPU 检查；旧 246 维位置策略不能加载为新位姿策略。原位置验收数值不变，朝向验收阈值仍待确定。

当前工作目录为 `artifacts/runs/diagnostic_pose_learning/`。`eval_initial8/operator_summary.json` 记录 fresh 初始化策略的 **8×20 秒 PhysX 批量前测，墙钟 49.73 秒**；全部完成且无跌倒，移动引用仍有约 38.5 cm 平均位置 RMSE。这是工程前测，不是学习结果或硬件有效性证明。

fresh 276-input、single-18 Actor 的 **1000 次迭代训练已完成，退出码 0**：`train_pose_config.json` 使用 4096 环境、24 步、5 epoch、4 minibatch、stage 1；现有 family 采样混入 `references/train/` 的 8 条 EE pose 引用。`train_initial_checkpoint.pt` 的腿 action std 为 .3（.2 rad scale 下等于 .06 rad），臂目标角 std 为 .01 rad；辅助预测／速度估计、自适应采样和 DR 均关闭。`train_pose1000_summary.json` 记录 98,304,000 transitions、20,000 次优化器更新，训练循环 3434.39 秒，有限性检查全部通过；累计 **1,780 次跌倒、98,519 次重置、29,739／17,694,720,000 个子步关节样本力矩饱和**，不能用后测零跌倒覆盖这些训练事件。训练入口沿用 `scripts/train.py --initialize-from`；已完成输出由唯一 GPU operator 管理，勿重复启动。

PhysX 后测有效结果为 `eval_post1000_openblas1/report.json`，与前测的比较见 `paired_pose1000.json`：前后均完整 **8×20 秒、零跌倒**，墙钟分别 49.73／50.07 秒。局部4例位置 RMSE 均值 **.02013 → .01557 m**、朝向 **.08539 → .15053 rad**；移动4例位置 **.38503 → .06573 m**、朝向 **.09385 → .28847 rad**。全部8例位置改善、全部8例朝向退步；总体位置 **.20258 → .04065 m**、朝向 **.08962 → .21950 rad**。局部／移动平均基座平面位移由 **.00469／.00452 m** 增至 **.05426／.11932 m**；位移增加本身不是成功标准。结果支持位置跟踪改善，不能宣布完整位姿跟踪成功。已授权朝向奖励权重对照，控制组训练已启动，候选组及评估结果待完成。

本轮对照检验：相同追加训练预算下，将 `orientation_tracking` 从控制组 **1** 提高至 **4**，能否改善朝向误差，以及位置误差和跌倒如何变化。两组均从 `checkpoint_000999.pt` 使用 `--initialize-from`、fresh Adam、学习率 `1e-5`、seed 0 开始，各训练 **250轮＝24,576,000 transitions／5,000次优化器更新**；其余数据、配置和物理参数固定。唯一 operator 顺序执行每组训练及 PhysX test8，再进行两组 MuJoCo test8；输出位于 `artifacts/runs/diagnostic_pose_learning/orientation_weight_comparison/`。尚无该对照结果，不新增朝向验收阈值。

首次 PhysX 后测退出码 **139**，`eval_post1000_console.log` 保留 OpenBLAS shutdown／fork 原生崩溃；仅增加 `OPENBLAS_NUM_THREADS=1` 后在新目录 `eval_post1000_openblas1/` 重试，退出码 **0**。复算实际前后比较需显式选择该后测：`python artifacts/runs/diagnostic_pose_learning/compare_pose8.py --after artifacts/runs/diagnostic_pose_learning/eval_post1000_openblas1/report.json`（使用 `pawweaver-runtime`）。

`references/test/` 为独立的 8 项固定评估套件。引用源是 **synthetic_fk_reference**，不是实测 FastUMI；局部引用为固定根 FK，移动引用离线加约 .6 m 水平根位移后仅保存 EE pose，未证明足步动力学可执行。现有 demonstrations `.align(reset_tcp)` 只平移位置、保留世界朝向；评估直接使用冻结世界位姿。详见 [引用构造记录](../artifacts/runs/diagnostic_pose_learning/references/README.md)。`scripts/evaluate_isaac.py --num-envs 8` 可消费该套件；旧两例 exact-hold 失败仍是历史结果，不要求批量动力学逐点复制顺序轨迹才可开展新任务评价。

同一最终策略的 MuJoCo 独立工程评估已完成，退出码 **0**，结果为 `eval_mujoco_post1000/report.json`：8例均完整20秒、零跌倒，平均位置 RMSE **.05230 m**、朝向 **.21391 rad**。`cross_engine_post1000.json` 比较命令退出码 **0**，PhysX／MuJoCo 现有仅位置 tracking pass rate 为 **6/8／5/8**；这不是完整位姿验收。下面保留已执行的 MuJoCo 命令；复现时另选新目录。诊断入口在模型编译前应用临时被动参数、力矩限值和 bundle 中的初始根高度，保留 `trained=false`。当前 test8 均为动态轨迹，跨引擎报告的静态到达率降幅与其是否达标均为 `null`，不填充为零；临时参数、少量合成引用和单种子结果不建立硬件有效性或正式验收。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python scripts/evaluate_mujoco.py --asset assets/generated/diagnostic --bundle artifacts/runs/diagnostic_pose_learning/train_pose1000/bundle --suite artifacts/runs/diagnostic_pose_learning/references/test --output artifacts/runs/diagnostic_pose_learning/eval_mujoco_post1000 --seed 0 --diagnostic --provisional-spec artifacts/runs/diagnostic_pose_learning/candidate_spec.json
```

以下保留资产／软件命令及旧位置任务历史复现记录；旧配置与 checkpoint 不作为本轮位姿训练入口。不启动独立 AS2／Piper-H 预训练，也不继续已取消的六轴位置候选。

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

## 旧位置任务：近距离学习与 4096 环境历史

`artifacts/runs/diagnostic_near_goal_learning_seed0/README.md` 的 **Executed result** 记录已完成的 32 环境 × 256 步 × 20 次 PPO（163,840 transitions、80 次优化器更新），以及同一冻结 16 例的顺序前后评估。实际前测是修复后的 `eval_pre_retry/results.json`，不是首次失败的 `eval_pre/results.json`；后测是 `eval_post20/results.json`。`paired20_summary.json` 显示 2 秒后的平均逐例 RMSE 从 0.0630698 降至 0.0624163 m，但仅 5 例改善、11 例变差，四条线轨迹全部变差，连续 2 秒处于 5 cm 内的案例从 9 减至 8。这是有效的负面学习结果，不能由总体均值的小幅改善宣布跟踪成功。前后各 16×20 秒均完整、有限，无跌倒、动作裁剪或力矩饱和；非零 goal-swap 响应只说明输出开始受目标影响。包仍为 `trained=false`。

首次前测因 `inference_mode` 创建的 PD target 在该上下文外被原位修改而失败；artifact 评估器改用 `no_grad` 后完成重跑，并精确复现第一例。另已纠正报告中的完整 canonical bundle hash。失败脚本、日志和 hash 修正前报告均保留，生产控制器没有因此修改。`artifacts/runs/diagnostic_batched_eval_probe/initial_two_comparison.json` 的两例批量对照虽满足状态/误差容差，但两例的连续 5 cm hold 时长均未精确一致，因此 `passed=false`；未放宽标准，也未进行 16 例批量评估。该旧实验的有效学习比较来自当时的顺序评估；当前位姿评估使用上文的新批量入口。

4096 配置和历史容量命令见 `artifacts/runs/diagnostic_4096_scale/README.md`；以下记录替代其中早期直接追加 997 次的建议。它只将上述训练配置改为 4096 环境、24 步 rollout、5 epoch、4 minibatch，复用完全相同的 `artifacts/runs/diagnostic_near_goal_learning_seed0/candidate_spec.json`，保持单个 18 关节 Actor、任务、奖励、2 ms/50 Hz、stage 0 和其他选项。`capacity3_summary.json` 已记录真实容量运行 exit 0：3 次迭代、294,912 transitions、60 次优化器更新，全部有限，零跌倒，53,084,160 个子步关节样本零饱和。吞吐约 29.8k–32.3k 环境控制步/秒；500 ms 采样的设备显存峰值 6,652 MiB 包含后台占用，也可能漏掉更短峰值。每环境仅 1.44 秒，不能据此认定长期学习或跟踪有效。

原 200 m 有限地面无法覆盖 8 m 间距下 1024/4096 环境的 ±124/±252 m 原点范围。修复 `a2076da` 按 cloner 网格跨度加两侧各 100 m 计算宽度；1/32/1024/4096 环境分别为 200/240/448/704 m。`cpu_coverage.json` 用安装版本 cloner 与 canonical 足部碰撞球验证重置几何，最小足部边缘余量约 99.754 m，地面材质、高度和厚度未变。CPU XY 布局证据与真实容量步进证据分别保留；运行 metadata 没有保存实际 XY 原点或地面宽度，不外推动态覆盖。

容量三轮 KL 为 42.725、0.0693、0.0358，保存的 Adam 学习率已到 `1e-5`。`artifacts/runs/diagnostic_4096_scale/kl_probe/README.md` 的固定观测判别支持 fresh Adam／`.001` 学习率重启对窄 Gaussian std 的敏感性，normalizer 变化也有影响；未发现 Actor/Critic/normalizer 加载错误，未保存真实首 minibatch，不能精确重建其因果。该历史续训选择已有 `--resume` 保留 Adam 和低学习率，没有再次初始化。

首次 `train50` 恢复在新增任何迭代前失败：`map_location` 将 CUDA RNG 状态搬到 GPU，但恢复 API 需要 CPU ByteTensor。最小修复 `f114ae5` 仅将 RNG 状态转回 CPU，5 项测试含真实 CUDA，实际 checkpoint 重放和独立 review 通过。失败日志及验证见 `train50_console.log` 和 `kl_probe/resume_actual_checkpoint.json`。修复后的 `train100/` 已用下列命令从 capacity iteration 2 严格恢复并完成 97 个追加迭代，达到总计 100 次；不重复启动同一目录。仿真回合按既有语义重新开始。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lyb/miniconda3/envs/pawweaver-train/bin/python scripts/train.py --asset assets/generated/diagnostic --diagnostic --provisional-spec artifacts/runs/diagnostic_near_goal_learning_seed0/candidate_spec.json --config artifacts/runs/diagnostic_4096_scale/candidate_training.json --resume artifacts/runs/diagnostic_4096_scale/capacity3/checkpoint_000002.pt --output artifacts/runs/diagnostic_4096_scale/train100 --seed 0 --num-envs 4096 --iterations 97 --headless
```

4096 阶段总计 9,830,400 transitions、2,000 次优化；97 轮续训全部有限，零跌倒/饱和、8,192 次超时重置，最终学习率自行恢复至约 `.000256289`。`checkpoint_000099.pt` 及 bundle 已在 runtime 独立校验。原冻结 16×20 秒顺序 `eval_post100/` 也已完成，配对结果见 `artifacts/runs/diagnostic_4096_scale/paired100_summary.json`：初始化／train20／train100 的 2 秒后平均逐例 RMSE 为 **6.307／6.242／6.844 cm**，四条线轨迹均退步，线轨迹平均为 **2.722／3.199／4.653 cm**。train100 相对初始化 6 好 10 差，相对 train20 5 好 11 差；连续至少 2 秒处于 5 cm 内的案例为 9/16。所有案例完整 20 秒，无跌倒、50 Hz 非足部净力超过 5 N、动作裁剪或子步力矩饱和。非零目标响应不能抵消任务指标退步，尚未学会有效跟踪；未追加 1000 次训练或改用批量评估。

旧位置任务的 balanced-y 实验与完整固定 16×20 秒评估已完成，见 `artifacts/runs/diagnostic_balanced_axis_learning/full16_summary.json`。2 秒后平均逐例 RMSE 为 **5.466 cm**，相对初始化 **6.307 cm** 有 9 好 7 差，相对 train100 **6.844 cm** 有 13 好 3 差；连续至少 2 秒处于 5 cm 内为 **10/16**。线轨迹均值 **3.251 cm** 仍高于初始化 **2.722 cm**，其中 3/4 变差。全部完整 20 秒，无跌倒、50 Hz 非足部净力超过 5 N、裁剪或子步饱和。这支持旧位置任务的部分改善，不是稳定全任务跟踪，更不验证新增的末端朝向要求。历史配置、日志和评价数值保留；六轴候选未启动并已取消。

上述记录均为旧位置任务历史；当前位姿接口及训练状态见本文开头。`artifacts/runs/subsystem_pretraining_transfer/` 随机教师 CPU 原型仅备用，无独立预训练或技能迁移证据。所有 GPU 实验仍由唯一 operator 顺序执行，重复实验使用新输出路径。临时硬件参数仍不能作为正式硬件有效性证据。

## 正式训练与课程

以下是待硬件 M0 闭合后的正式流程模板；新位姿接口已经实现，但当前工程包和临时参数不能替代正式训练前提：

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

在 `pawweaver-data` 中运行 `python -m pawweaver.data --help`。输入需要 pose、逐帧时间戳、稳定 source ID，以及单位、sensor→TCP、source→task、速度/加速度、工作区边界的配置。转换器组合完整 source→task、输入 pose、sensor→TCP 刚体变换；位置和 WXYZ 朝向使用相同时间缩放，朝向以 SLERP 重采样。转换器不下载视频，增强前按 source ID 划分数据。当前没有下载实测 FastUMI 示范集，训练引用为上文合成 FK 数据。

在 `pawweaver-runtime` 中运行 `python -m pawweaver.visual_runtime --help`。需要有效资产、策略包、轨迹与 scenario JSON。scenario 定义 `marker_id`、`marker_size_m`、`marker_to_goal`（标记系平移）、`marker_to_goal_quat_wxyz`（显式标记到目标朝向标定），可加入延迟、遮挡、深度缺失、位置噪声和随机种子；`--record` 输出腕部视频。

相机在 2 ms 物理时钟上调度约 30 Hz 图像，50 Hz 控制器消费到达的测量。真值仅用于场景生成和评分。持续失跟后暂停参考推进、保持有界关节目标；这种保持的整机稳定性仍需训练策略验证。基座定位来自仿真状态，不包含真机自主定位。
