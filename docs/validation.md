# 验证记录 · 2026-09-10

这些结果区分组合机器人模型检查与合成机器人软件检查。**已完成有界 PPO 学习实验，尚无通过验收的 AS2 + Piper-H 到达、连续跟踪或视觉控制策略。** M0 未闭合；M1–M4 的性能验收还没有完成。

**当前新增证据**：[广范围6D训练](../artifacts/runs/diagnostic_pose_learning/wbc_random_training/README.md) 的2000轮及双引擎十项后测均已实际退出0。训练196,608,000 transitions／40,000次更新，全部有限，累计22,938跌倒／76,267重置；这是工程训练完成，尚非任务达标。

| 最终策略测试 | PhysX | MuJoCo |
|---|---|---|
| 独立采样64例，各请求60秒 | 位置通过11/64；跌倒24/64，完整40/64 | 位置通过12/64；跌倒36/64，完整28/64 |
| 静态到达16例，各请求12秒 | 位置到达3/16；跌倒13/16 | 位置到达2/16；跌倒14/16 |
| 设计工作空间4例 | 完整3/4；far倾斜跌倒 | 完整4/4；far完成60秒 |
| 历史test8／far_return60回归 | test8跌倒1/8；far跌倒 | test8跌倒1/8；far跌倒 |

上表位置到达沿用10秒内进入5cm并保持1秒的条件，位置跟踪沿用原报告条件；朝向验收仍未定义，不能称完整位姿通过。设计工作空间4例附近的几何见证用于构造训练分布，不是独立holdout；静态16例与holdout64共享测试几何池，不能宣称两者互相独立。MuJoCo设计workspace far完成60秒，不等于历史far_return60成功，两者必须按suite区分。原报告每例误差使用其实际存活时长，提前终止后的短均值不能直接与完整60秒前测比较。具体数值、终止原因和原报告索引见 [evaluation_summary.json](../artifacts/runs/diagnostic_pose_learning/wbc_random_training/evaluation_summary.json)。

[运动读出](../artifacts/runs/diagnostic_pose_learning/wbc_random_training/motion_readout/README.md) 显示臂、身体和腿共同参与，但长期高抬单足、三足承载和倾倒仍未解决；局部任务收益不能替代用户要求的连贯、稳定全身动作。[支撑配方候选](../artifacts/runs/diagnostic_pose_learning/wbc_support_learning/README.md)因课程放宽后仍持续退化，已在记录336轮后提前停止；最新可用策略为301轮检查点，未包含后35轮更新。新双引擎后测正在执行。训练末50轮非终止奖励裁零99.872%、位置／朝向均误差.25451m／1.44972rad；这是训练退化证据，不是独立任务得分或1000轮必败证明。单种子、临时硬件参数、`trained=false`及以下原验收边界保持不变。

**历史UMI-inspired证据**：[UMI-inspired 训练方案](../artifacts/runs/diagnostic_pose_learning/umi_recipe/README.md) 的1000轮与四项双引擎后测已全部实际退出0。训练98,304,000 transitions／20,000次更新、全部有限，累计13,865跌倒／42,092重置／1,248,573非足碰撞控制样本；力矩饱和16,918,732／17,694,720,000子步关节样本。两引擎local4均完整20秒，位置／朝向RMSE均值为PhysX .006117m／.010725rad、MuJoCo .003630m／.010436rad，均优于既有背景控制；moving均3/4提前跌倒，far分别6.68秒／1.64秒跌倒。当前仅支持局部精度改善，不支持移动／远距任务成功或完整WBC。逐例比较使用共同2秒至较早终点窗口；MuJoCo far无post-2秒共同窗口，明确null。原标准如下，保持不变。

用户最新目标是**同一个 18 关节强化学习 Actor 仅跟踪世界系末端位置、朝向轨迹**，输出 12 腿＋6 臂统一关节位置目标，并自主协调底盘运动，无外部底盘速度命令。参考 MLM 的因果目标历史和可选预测未来目标，后续面向手持 UMI 遥控；基座速度仅作估计量或 Critic／训练标签。当前已实现 276 维位姿观测及 schema-2 数据／策略接口；旧 246 维 position-only 策略缺少朝向目标，不能作为新接口策略。以下旧实验和原验收数值保留在其原有证据范围内，不能改称完整位姿目标的验证；新增朝向误差验收阈值尚未确定。

新固定组过滤前的真实组合模型 PhysX 接触读数、碰撞奖励及含接触量的 Critic 观测受到已确认的内部碰撞影响；那些有限 PPO 更新仍是软件集成证据，不能作为接触行为正确性的证据。下文保留历史结果并标出当前修正。

## 原计划验收标准与当前证据边界

以下保留原计划第 5 节标准，区分正式验收与当前工程筛选；不新增“每例所有指标都改善”的硬门槛。

| 项目 | 原计划标准 | 当前工程证据边界 |
|---|---|---|
| 静态到达 | 对水平距离约 0.3–2 m 的目标先做可达性筛选；10 秒内进入 5 cm 误差范围并保持 1 秒，成功率 ≥95%。 | 少量合成目标或动态 test8 的位置统计不等于该静态到达验收。 |
| 连续跟踪 | 目标速度 ≤0.3 m/s，完整运行 60 秒；过渡后位置 RMSE ≤8 cm、P95 ≤15 cm，至少 95% 回合无跌倒。 | 当前单种子、临时参数下的 test8×20 秒及 far60 用于筛选；存活或短时低误差不能替代完整任务指标。 |
| 重复性与跨引擎 | Isaac Lab、MuJoCo 各至少 100 回合，主要结果覆盖 3 个训练种子；MuJoCo 静态到达成功率下降 ≤10 个百分点。 | 当前有限回合、单种子双引擎结果尚不满足该规模；不将不同时长的全程误差直接配对。 |
| 全身协同 | 必须包含需要迈步的目标，同步记录基座位移、腿臂运动、足部接触与 TCP 误差，提供协同证据。 | 仅有基座位移、足端移动或单次固定腿输入，不能证明受控迈步与有效协同。 |
| 消融 | 比较预测开关、均匀与自适应采样、合成与示范数据的收益。 | 各次有界对照只支持其实际检验的因素与结果，不替代完整消融。 |
| 视觉闭环 | 单独报告测量误差、失跟与恢复，不混同策略使用的视觉测量和状态真值。 | 状态真值跟踪结果不能代替视觉闭环；既有短时视觉证据保留其原有范围。 |

当前代码的误差统计使用 2 秒过渡段；不足 2 秒的报告保留原短时序统计，不能视为完整任务成功。新增朝向正式验收阈值仍待用户确定，奖励宽度 `0.5 rad` 不是验收阈值。正式 M0 参数仍未闭合，临时硬件参数下的工程结果不能建立硬件有效性或正式里程碑完成。

**全身协同的场景含义（用户明确，2026-09-10）**：除需要迈步的远目标，还须覆盖低位、高位、侧向与朝向变化，按任务需要展示腿部支撑、身体升降／倾斜和臂运动共同扩大可操作空间。单Actor输出18关节只是架构条件；混合训练平均误差、水平跟踪或单纯身体下沉不证明这一能力。历史窄分布实验的静态目标dz±.08m、大多数合成轨迹固定reset朝向及小高度／朝向变化的训练引用，未覆盖该要求；有Z变化的smooth轨迹不能被说成完全平面，也不能替代完整场景验证。历史默认基座z<.2m终止条件与.18m四足支撑几何见证冲突；上述新实验采用.15m且保留base-up-Z .35，须继续依据实际接触和任务证据区分正常低姿态与倒地，不能以改阈值制造通过。这里澄清原能力目标，不新增数值通过阈值、不改变上表原标准。

## 当前位姿路径与 KMA 实际反馈

位姿合同、四元数归一化／符号等价、因果历史、完整 TCP 固定变换、数据 SLERP、视觉旋转标定及批量参考时钟已有 CPU 检查；fresh 276-input Actor／386-input Critic 的严格加载、eager/JIT 和独立 runtime 输出一致，见 `artifacts/runs/diagnostic_pose_learning/initialization_verification.json` 与 `runtime_verification.json`。这些证据验证实现与接口，不证明已学会跟踪；早期依赖缺失测试失败及修正日志保留在该目录 README。

`references/` 有 8 train＋8 test 的 20 秒合成 FK 位姿引用。16,016 个采样的 canonical FK／关节范围检查和 80 个静态 MuJoCo 关键点通过，后者无非足部接触；移动参考只属名义运动学，非完整动力学可达性证明。`eval_initial8/operator_summary.json` 已记录新初始化策略的真实 **8×20 秒批量 PhysX**：退出码 0、墙钟 **49.7279 s**、8 例完整无跌倒。局部／移动平均位置 RMSE 分别 **.02013／.38503 m**，朝向 RMSE 分别 **.08539／.09385 rad**；最大基座位移均不足 .005 m，移动跟踪明显不足。这是 `trained=false` 前测，不能宣称新任务学习有效。

**1000 次迭代训练已完成，operator 确认退出码 0**，使用 fresh 276-input、stage 1 现有 family 混采 8 条训练引用，独立 test8 不进入训练；腿 std .3 action＝.06 rad，臂 .01 rad，辅助模块与 DR 关闭。`train_pose1000_summary.json` 记录 4096 环境×24步×1000轮＝**98,304,000 transitions、20,000 次优化器更新**，训练循环 **3434.39 秒**，有限性检查全部通过。训练累计 **1,780 次跌倒、98,519 次重置，29,739／17,694,720,000 个子步关节样本力矩饱和**；这些是变化中的训练样本统计，不是独立测试结果。朝向奖励起始参数属于工程选择，正式朝向验收阈值未设定。

`paired_pose1000.json` 使用真实前测 `eval_initial8/report.json` 与有效后测 `eval_post1000_openblas1/report.json`，2秒后逐例 RMSE 的组均值如下。前后各 **8×20秒均完整、零跌倒**，墙钟分别 **49.7279／50.0695秒**；此处零跌倒仅指评估，不否定上述训练事件。

| 独立测试组 | 位置 RMSE 前→后（m） | 朝向 RMSE 前→后（rad） | 平均基座平面位移前→后（m） |
|---|---|---|---|
| local4 | .020135 → .015566 | .085395 → .150531 | .004689 → .054257 |
| moving4 | .385032 → .065727 | .093854 → .288474 | .004523 → .119320 |
| all8 | .202583 → .040647 | .089625 → .219503 | .004606 → .086788 |

位置 **8例改善、0例变差**，朝向 **0例改善、8例变差**；两组各为位置4好、朝向4差。位置跟踪已显示学习改善，但不能由位置均值改善或基座位移增加宣布完整位姿任务成功。后续朝向奖励权重对照已完成，见下；不新增朝向验收阈值。

<a id="pose1000-foot-motion"></a>

独立 reviewer 复核 `eval_post1000_openblas1/` 已保存的50 Hz足力与观测，以足力 **>1 N** 描述采样时刻的承重接触，并从观测、世界TCP四元数恢复基座旋转和关节角后进行FK足端重建；重建TCP残差 **0.7–3.3 µm** 支持该重建路径。下表保留原数值；复核确认其足力分类复用了历史 `initialization_verification.json` 的body_names顺序（FR／FL／RR／RL列为17／16／19／18），该批trace未记录实际runtime顺序。因此承重／接触标签依赖这一历史顺序假设；FK足位置和基座位姿结论独立。1 N仅为描述阈值，不是验收标准。

| PhysX移动案例 | 50 Hz足力与重建足端水平运动 | 最低基座z（m） | 基座倾斜 |
|---|---|---|---|
| moving00 | 所有记录时刻四足接触；RL承重水平路程约13.0 cm | .291 | 最大roll 5.9° |
| moving01 | FR约20 ms、RR/RL各约60 ms低于阈值；RR承重水平路程约12.8 cm | .263 | 最大roll 11.5° |
| moving02 | FL约60 ms低于阈值；FL净水平位移约13.7 cm | .248 | 最大pitch 9.2° |
| moving03 | 所有记录时刻四足接触；FL承重水平路程约13.2 cm | .272 | 最大roll 7.4° |

这些记录独立支持下沉、倾斜和足端移动；承重拖移的解释受上述标签假设限制，尚未证明交替迈步；球形足滚动的贡献未分离，不能把50 Hz接触记录扩展为每个物理子步的结论。MuJoCo 同批后测未保存接触，不能据此确定步态或滑移；其 moving02 最低基座z约 **.229 m**、pitch约 **−11.9°**，FK重建TCP残差最大 **.87 mm**，不依据微小足高差断言离地。

后续已成功执行的CPU判别复用既有 `candidate_spec.json`、`diagnostic_active_arm_unfold/plan.json` 的 `actions[1]`、保存的trace观测和 `checkpoint_000999.pt`。在现有腿部 `q0 ± .2 rad` 范围内，每腿11³网格找到抬脚见证：FR/RR为 **[-.2, .65, -1.85] rad**，FL/RL为 **[+.2, .65, -1.85] rad**，足link相对q0升高 **68.36 mm**、相对initial hold约 **100–102 mm**。这仅是固定基座FK，未经碰撞筛选、不是全局极值，也不证明动态抬脚。用train1000 bundle回放8例保存观测，`clip(raw_action)` 与保存action最大差 **1.32e-6**；moving00–03在 `t >= 2 s` 的已裁剪腿动作样本处于边界的比例分别 **81.44／80.85／88.83／91.71%**，各例原始动作绝对值中位数为 **2.00–2.44**、最大值分别 **3.76／4.99／3.52／3.57**。

checkpoint实际腿部标准差为 **.331–.855**；对4个moving案例的 **3,604个保存状态×12腿维**，按各维Normal分布计算 `P(-1<a<1) = Φ((1−μ)/σ) − Φ((-1−μ)/σ)`，边际概率均值 **16.66%**、中位数 **2.211%**。仅在数学上将σ乘3的反事实，均值为 **24.29%**、中位数 **20.49%**；这不是联合动作进入区间的动态频率，也不证明学习收益。现有动作范围并非几何上绝对禁止抬脚，均值越界及有效探索值得后续检验；尚未据此选定新实验，当前权重对照保持不变。

朝向权重对照已完成，问题是在相同追加训练预算下，`orientation_tracking=4` 相对控制组 `1` 是否改善朝向，并观察位置误差和跌倒。两组均从 `checkpoint_000999.pt` 经 `--initialize-from` 初始化，fresh Adam、初始学习率 `1e-5`、seed 0，各 **250轮、24,576,000 transitions、5,000次优化器更新**。`orientation_weight_comparison/` 的两份config仅朝向权重不同；实际run的source哈希、模型、临时参数和数据一致，Git commit/dirty描述不同。各 `metrics.jsonl` 均有迭代0–249，有限性检查全通过，末检查点为 `checkpoint_000249.pt`。README记录两组训练及四批评估退出码均为0；本次两组训练／评估已结束，未追加这两组训练。

训练循环控制／候选为 **842.79／840.75秒**；跌倒 **217／240次**、重置均 **24,581次**，饱和 **88,745／276,114** 个样本，每组 **4,423,680,000** 个关节物理子步样本。这些实测训练事件保留，与独立测试统计分开。

| 引擎／测试组 | 位置RMSE 控制→权重4（m） | 朝向RMSE 控制→权重4（rad） |
|---|---|---|
| PhysX local4 | .015082 → .032850 | .137245 → .095737 |
| PhysX moving4 | .055770 → .079619 | .198901 → .143393 |
| PhysX all8 | .035426 → .056234 | .168073 → .119565 |
| MuJoCo local4 | .041482 → .033197 | .186228 → .124606 |
| MuJoCo moving4 | .083072 → .086986 | .268551 → .150182 |
| MuJoCo all8 | .062277 → .060091 | .227390 → .137394 |

四批评估各8例均完整20秒、零跌倒。PhysX朝向8好、位置8差；MuJoCo朝向8好、位置4好4差。仅位置tracking pass为PhysX **7/8 → 6/8**、MuJoCo **5/8 → 5/8**；`pose_acceptance_passed=null`，不将其称为位姿验收。证据支持朝向改善跨引擎保留，同时存在PhysX位置代价；控制组继续训练本身已改善1000轮PhysX结果，权重效应应依据直接控制→候选对照。单种子、临时参数、合成引用仍不建立收敛、硬件有效性或正式里程碑完成。

直接PhysX比较见 `paired_control_candidate.json`，相对原1000轮比较见 `paired_before_control.json`／`paired_before_candidate.json`，两组跨引擎比较见 `cross_engine_control.json`／`cross_engine_candidate.json`；MuJoCo组间数值按两份原始 `eval_mujoco_*/report.json` 的同名案例复算。将PhysX专用 `compare_pose8.py` 用于MuJoCo组间报告曾被“expected world_tcp_pose PhysX report”拒绝，空输出保留为 `paired_mujoco_control_candidate.failed_empty.json`，不作为有效比较报告，未改脚本绕过边界。

此前建议的CPU配对回放已完成（退出码0）：复用 `orientation_weight_comparison/` 两组bundle及 `eval_control/`、`eval_candidate/` 原始trace，`clip(raw)` 与保存action最大差 **1.55e-6**。在两组moving4的 `t >= 2 s` 保存状态上，腿raw动作越界比例 **90.28% → 90.87%**，臂 **4.79% → 3.59%**，臂均仅J5越界；基座倾斜变化混合，未见权重4新增普遍裁剪。两组都存在腿动作顶边，不能据此把位置代价单独归因于候选新增裁剪；观测与步后状态按一帧偏移对齐，TCP位置重建残差小于 **.281 µm**；足路径仅报未接触门控的FK运动，不新增承重／步态结论。这是保存状态关联证据，不是奖励或探索的动态因果证明。 评估器已在8059924中为新trace保存本次runtime的contact_body_names，3项CPU序列化测试通过；后续已完成的8份PhysX trace已验证实际输出，见下。旧trace没有补填。

**联合位姿奖励候选已完成训练及双引擎评估，均退出0**：`2*rpos+rrot → 3*rpos*rrot` 保持 `.15 m／.5 rad` 宽度和峰值3，但不保持梯度；源代码 `d26b514` 已review、12项CPU测试通过。复用w1控制组，从同一原 `checkpoint_000999.pt` 经fresh Adam、初始LR `1e-5`、seed0完成 **250轮／24,576,000 transitions／5,000次更新**，其余配置固定。训练循环 **845.19秒**，全部有限；**297次跌倒、24,589次重置、110,112／4,423,680,000** 个子步关节样本饱和，不能由后测零跌倒覆盖。

PhysX控制→联合奖励位置RMSE **.035426 → .038529 m**（2好6差）、朝向 **.168073 → .128677 rad**（8好）；MuJoCo位置 **.062277 → .037347 m**、朝向 **.227390 → .159585 rad**（各7好1差）。候选两引擎各8×20秒完整、零跌倒，现有仅位置pass均7/8，位姿验收仍为null。相较权重4，联合奖励的PhysX位置代价较小，但朝向改善也较少，尚未实现各例两类误差同时改善。训练、逐例／分组比较与复现命令见 [coupled_pose_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/coupled_pose_comparison/README.md)。 分组控制→候选：PhysX local4位置 **.015082→.022776 m**、朝向 **.137245→.094756 rad**，moving4 **.055770→.054282 m／.198901→.162598 rad**；MuJoCo local4 **.041482→.013909 m／.186228→.119942 rad**，moving4 **.083072→.060786 m／.268551→.199227 rad**。依据为同目录 `train_summary.json`、`paired_physx_control_candidate.json`、`paired_mujoco_control_candidate.json` 和 `cross_engine_candidate.json`；仍为单种子临时参数工程证据，`trained=false`。

新候选8份PhysX trace实际保存30个Unicode `contact_body_names`，与30列足／身体净接触力对齐，`allow_pickle=False`可读，见 `coupled_pose_comparison/contact_names_check.json`。列名直接来自本次runtime；这验证了实际保存路径，不能追认旧trace的body顺序，也不证明接触对动力学或步态。

腿部初始std对照的两组250轮及双引擎test8已完成，均退出0：共同起点为coupled250最终checkpoint，fresh Adam／初始LR `1e-5`／seed0，仅候选初始12腿std×3，臂和其他状态不变，std继续可训练。每组 **24,576,000 transitions／5,000次更新**、全部有限；控制／候选训练 **852.32／861.51秒**、跌倒 **567／736**、重置 **24,624／24,655**、饱和 **134,135／283,597**（分母各 **4,423,680,000**）。PhysX位置RMSE **.032482→.033158 m**（4好4差）、朝向 **.131306→.117076 rad**（6好2差）；MuJoCo位置 **.032019→.038960 m**（2好6差）、朝向 **.128080→.149437 rad**（8例全差）。四批均完整8×20秒、零跌倒，但朝向收益未跨引擎保留，当前不支持采用std×3。详细配置及逐例证据见 [leg_std_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/leg_std_comparison/README.md) 和同目录两份训练summary、`paired_*_control_candidate.json`。

扩展远目标 `far_return60/mujoco_coupled250/` 已完成，进程退出0但任务在 **22.94／60秒** 因倾斜跌倒终止；截至终止的RMSE为 **.396744 m／.298468 rad**，不能视为完整60秒指标。std远目标补测已结束，随后选择原std控制组checkpoint进入 [sustained_learning_next](../artifacts/runs/diagnostic_pose_learning/sustained_learning_next/README.md)：同时改变stage1→2和回合20→60秒、其余固定，fresh Adam／初始LR `1e-5`／seed0。训练250轮和四项后测均退出0；**24,576,000 transitions／5,000次更新／829.83秒**、全部有限，训练 **1,427跌倒／8,479重置／4,467,120÷4,423,680,000子步关节样本饱和**。其中 **7,052次非跌倒60秒超时**；现有事件计数给出20秒后全局transition下界 **14,104,000**，不是每个环境／family的覆盖或学习收益证明。

新策略far两引擎均完整60秒、无跌倒，但PhysX／MuJoCo位置RMSE仍 **.621440／.621054 m**，基座最大水平偏移仅 **.178279／.155302 m**，不能把存活延长称为远目标跟踪成功。与所选std控制组的共同时间窗比较，PhysX位置 **.157260→.161315 m**（2–17.38秒）、朝向 **.162577→.076339 rad**；MuJoCo位置 **.516660→.582251 m**（2–28.18秒）、朝向 **.586344→.176258 rad**。位置没有改善，朝向改善；不直接比较不同时长的全程RMSE。

固定test8仍各8×20秒、零跌倒：PhysX位置 **.032482→.035531 m**（local4全好、moving4全差），MuJoCo **.032019→.045159 m**（8例全差）；朝向分别 **.131306→.094178 rad／.128080→.099021 rad**，两引擎均8例全好。依据为同目录 `train250_summary.json`、`execution_status.json`、`paired_far60.json`、`paired_physx_test8.json`、`paired_mujoco_test8.json`。

位置奖励宽度 **.15→.45 m** 对照已完成，复用原stage2控制组；候选250轮 **24,576,000 transitions／5,000次更新／827.20秒**，退出0、全部有限。训练 **1,422跌倒／8,471重置／4,053,721÷4,423,680,000子步关节样本饱和**，其中 **7,049次非跌倒60秒超时**。test8两引擎均完整8×20秒、零跌倒：PhysX位置 **.035531→.063010 m**（8例全差）、朝向 **.094178→.076464 rad**（6好2差）；MuJoCo位置 **.045159→.069698 m**（1好7差）、朝向 **.099021→.090498 rad**（6好2差）。

PhysX far两组均60秒无跌倒，位置 **.621440→.634521 m**变差、朝向 **.269527→.169957 rad**改善；MuJoCo候选在 **8.24秒跌倒**，控制组完整60秒。共同 **2–8.24秒** 窗口的位置 **.023773→.151065 m**、朝向 **.074446→.246053 rad**均退步，不能用候选短程均值与控制完整60秒均值比较。依据见 [position_width_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/position_width_comparison/README.md) 及同目录 `train_candidate_summary.json`、`paired_far60.json`、两份 `paired_*_test8.json`。**不采用.45，默认.15未改**；width透传参数保留用于已完成实验复现。

腿均值越界正则 **.001** 候选已完成250轮及双引擎test8／far60四项后测，复用原stage2控制组，五个仿真进程均退出0。训练 **24,576,000 transitions／5,000次更新／831.81秒**，全部有限；实际加权正则在250轮均非零。test8均完整8×20秒无跌倒，结果混合：PhysX位置／朝向RMSE **.035531→.036887 m／.094178→.088722 rad**，MuJoCo **.045159→.041206 m／.099021→.104356 rad**。far两组两引擎均完整60秒无跌倒，共同2–60秒位置／朝向RMSE为PhysX **.621440→.677840 m／.269527→.302050 rad**、MuJoCo **.621054→.644799 m／.173467→.183270 rad**，均退步。**不采用.001，默认系数0及位置奖励宽度.15未改**。完整命令、退出码、逐例结果及原始失败记录见 [leg_mean_bound_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/leg_mean_bound_comparison/README.md)。

[冻结状态CPU回放](../artifacts/runs/diagnostic_pose_learning/leg_mean_bound_comparison/response_probe/README.md) 对两策略使用相同状态及沿TCP到目标方向的±5 cm历史平移，经过实际JointPD映射：四个远状态库的均值越界幅度均降低，但两策略四个髋关节均无实际响应，后腿未恢复持续响应。此结论限于该方向、该远状态子集，是条件响应证据，不证明动态因果或更大系数有效。

progress权重 **1→10** 的同起点250轮及四项后测已完成，复用stage2/60秒控制，五个仿真进程均退出0。训练 **24,576,000 transitions／5,000次更新／835.42秒**，全部有限，累计 **1,933次跌倒／8,674次重置**。test8按每例共同窗口比较：PhysX位置／朝向RMSE **.035265→.030710 m／.093109→.110703 rad**，位置7好1差、朝向2好6差；`test_moving_01`在 **16.44秒跌倒**，其余七例完成20秒。MuJoCo八例均完整20秒无跌倒，位置 **.045159→.027206 m**（8例全好），朝向 **.099021→.127383 rad**（2好6差）。两引擎local4朝向均全退步。

far控制两引擎均完整60秒，候选PhysX **26.62秒**、MuJoCo **33.62秒跌倒**。共同窗口分别为2–26.62秒／2–33.62秒，位置／朝向RMSE为PhysX **.521746→.503429 m／.214749→.135174 rad**、MuJoCo **.701016→.657175 m／.196216→.124050 rad**；两类误差虽均改善，却失去60秒存活，**当前候选不采用**。不能用候选短程和控制全程均值或改权重后的训练reward宣称进步。逐例、P95、共同窗及失败记录见 [progress_weight_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/README.md)。默认 **progress1、bound0、width.15** 未改；原验收及`trained=false`边界保留。

同配置经`--resume`追加 **750轮至累计1000轮** 及四项后测均已实际退出0，追加训练全部有限；这比较的是progress250→1000的额外学习预算，原progress1控制仅作不同预算背景。test8两引擎均八例完整20秒无跌倒；同例共同窗PhysX位置 **.030710→.018504 m**（8例全好），朝向 **.110703→.121581 rad**（3好5差）；MuJoCo位置 **.027206→.026832 m**（4好4差）、朝向 **.127383→.183435 rad**（1好7差），两引擎local4朝向均全退步。far跌倒时间 **PhysX26.62→10.90秒、MuJoCo33.62→12.70秒**；共同窗位置／朝向RMSE分别 **.011726→.062589 m／.080835→.129644 rad**（2–10.90秒）、**.036550→.096589 m／.079253→.276799 rad**（2–12.70秒），均变差，**不采用1000轮策略**。不能把更早终止后的短程均值当作全程改善，完整证据仍见上述progress对照记录。

[失稳过程回读](../artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/continuation_failure_analysis/README.md) 依据实际跌倒标志、高度和原终止判据识别倾斜触发；终止时刻的精确倾斜值未保存。PhysX后足力下降出现在快速倾倒前，但此顺序是观察证据，不能归因于奖励系数；MuJoCo未保存接触力。[短时冻结采集与回读](../artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/continuation_failure_analysis/early_training_signal_capture_preparation/readout/README.md) 已完成：2.4秒、64环境、7,680条记录、零更新，实际退出0并通过冻结检查；记录12次早期自然跌倒，均发生于0.84–1.86秒，属于startup/reset训练分布，不能当作far失稳重现。

[termination−5→−50对照](../artifacts/runs/diagnostic_pose_learning/termination_weight_comparison/README.md) 的同起点250轮及四项后测均已退出0，训练全部有限，**不采用该候选、不追加预算**。主对照仍为原progress10/termination−5的250轮。候选test8两引擎全8×20秒无跌倒；保留原PhysX moving01的2–16.44秒共同窗后，位置／朝向均值为PhysX **.030710→.031079 m／.110703→.109512 rad**、MuJoCo **.027206→.029417 m／.127383→.123521 rad**，表现混合，PhysX local4两类误差均全退步。

PhysX far由26.62秒跌倒恢复为完整60秒，但主共同窗2–26.62秒的位置／朝向RMSE **.503429→.504554 m／.135174→.153411 rad**均差；完整60秒位置RMSE仍 **.595804 m**，相较既有稳定progress1背景的.621440 m仅小幅降低，不能把存活当成跟踪成功。MuJoCo由33.62秒变为 **22.46秒更早跌倒**；共同窗2–22.46秒两类RMSE虽 **.367297→.350794 m／.126017→.114260 rad**，却不能抵消更早终止，朝向P95亦退步。逐例、P95、完整时长和主对照／稳定背景分别保留于实验记录。[已保存数据的只读判别](../artifacts/runs/diagnostic_pose_learning/termination_weight_comparison/next_learning_readout/README.md) 已完成CPU回放（退出0），termination−50未恢复持续髋／后腿有效目标响应，不证明裁剪是唯一原因。[softsign腿均值对照](../artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/README.md) 的250轮训练和四项后测均已实际退出0，**不采用候选**。复用原T50控制的起点、fresh Adam／初始LR `1e-5`／seed0／4096×24×5×4／stage2／60秒／progress10／termination−50，仅增加`leg_mean_transform=softsign`；原T50训练与后测直接复用。训练全部有限，累计841次跌倒／8,304次重置；test8两组两引擎均完整8×20秒无跌倒，PhysX位置／朝向各1好7差，MuJoCo位置6好2差但朝向总体均值退步。本轮T50 test8使用完整2–20秒窗口，不沿用旧T5比较的16.44秒窗口。PhysX far全60秒存活但位置RMSE **.595804→.619860 m**变差；MuJoCo **22.46→28.72秒仍跌倒**，共同2–22.46秒位置／朝向RMSE **.350794→.384550 m／.114260→.187805 rad**均变差。零更新时腿PD目标已改变，不能称为仅梯度修复。默认identity、原验收与`trained=false`保持不变，`3869dee`的opt-in实现保留复现。[学习后只读回放](../artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/post_learning_readout/README.md) 与 [动作能力证据核对](../artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/actuation_evidence/README.md) 已完成：softsign均值路径保留，保存远状态上髋关节高斯采样落入动作区间的解析概率约48%，但±5 cm目标扰动下RR thigh的PD目标响应仅约.00024 rad，未证明有效协调；现有加载证据支持站立／展开臂，尚无受控抬足证据。固定RR腿、当前动作范围的 [7秒双引擎同输入诊断](../artifacts/runs/diagnostic_loaded_leg_lift/README.md) 已完成：两项有效运行均退出0、全7秒有限且无跌倒／饱和，但RR仅短暂卸载，FL反而离地，机身倾斜约19°且回程未恢复，未实现干净受控抬放。MuJoCo直接记录RR大腿接地，PhysX仅提供相应大腿净力证据；不能由这一固定输入证明动作范围不足。前两次诊断脚本失败及修复均保留，该诊断未改默认参数，softsign仍不采用。

[adaptive sampling对照](../artifacts/runs/diagnostic_pose_learning/adaptive_sampling_comparison/README.md) 的250轮训练与四项固定后测均已实际退出0，**不采用候选、不追加预算或参数扫描**。唯一配置差异为`adaptive_sampling=false→true`，复用uniform softsign250同初始化、同预算控制；采样路径不同，不宣称逐轨迹配对。现有checkpoint确认概率与完成回合分布改变，采样并非未生效，但位置／跌倒EMA不含朝向／碰撞，不能替代位姿收益。test8两组两引擎均完整8×20秒无跌倒；共同2–20秒内PhysX朝向8例全退步，MuJoCo位置8例、朝向7例退步。PhysX far两组均60秒，位置RMSE仅.619860→.614141 m；MuJoCo由28.72秒跌倒变为完整60秒存活，但共同2–28.72秒位置RMSE **.572716→.598932 m**变差，候选完整60秒仍 **.621898 m**。存活改善不等于远目标跟踪成功，也不证明动作范围不足。默认配置、原验收与`trained=false`保持不变。

[新腿动作范围对照](../artifacts/runs/diagnostic_pose_learning/leg_range_comparison/README.md) 的新.2控制与新.5候选各250轮及四项固定后测，**十条命令均实际退出0，不采用.5，不自动追加预算或尺度扫描**。每组24,576,000 transitions／5,000次更新，全部有限；训练跌倒1,549→2,362，子步力矩饱和0.31045%→0.55579%。两组两引擎test8均完整20秒无跌倒，共同2–20秒平均位置／朝向RMSE：PhysX **.027977→.036483 m／.100327→.188344 rad**，MuJoCo **.026513→.031052 m／.107312→.161160 rad**；朝向各8例全退步。MuJoCo位置单项通过数7→8保留，不能称完整位姿成功。far两组两引擎均完整60秒无跌倒，共同2–60秒PhysX位置仅.636364→.628724 m小幅改善、朝向.180363→.186254 rad变差；MuJoCo位置／朝向 **.621293→.638153 m／.148586→.242929 rad**均变差。逐例P95、时长与全部训练事件见实验记录。

初始确定性腿PD目标及裁剪前角度std匹配，不代表加载站姿；动作范围、裁剪／探索、归一化动作历史、动作变化率奖励及优化坐标属于联合处理，历史T50／softsign未替代新控制。[保存轨迹回读](../artifacts/runs/diagnostic_pose_learning/leg_range_comparison/candidate_motion_readout/README.md) 显示候选两引擎2–30秒基座前进更少、最低高度更低；PhysX仅有50Hz RR足净力卸载记录，不能据此称有效步态或证明动作范围不足。[参考训练核对](../artifacts/runs/diagnostic_pose_learning/leg_range_comparison/reference_training_audit/README.md) 已完成：UMI官方默认支持无步行检查点的单18动作直接训练，但目标表达、裁剪／探索、课程及PPO配方不同；没有因此选定新训练机制或预算。.5仍是临时工程参数，默认、原验收与`trained=false`不变。纯位姿学习稳定前继续暂缓视觉扩展，没有新增60秒视觉结果。

原episode17另已派生20秒schema-2工程命令，CPU构造／加载检查及coupled250 MuJoCo评估均退出0；完整20秒、无跌倒，2秒后位置／朝向RMSE **.049607 m／.155485 rad**。使用人为 `20*i/119` 重定时、米制XYZW工程假设和一次固定 `G*T0^-1*Ti` 对齐；未知sensor→TCP外参仍未消除，不是原始时序或正式TCP示范转换，未加入当前训练或冻结test8。见 [派生命令](../artifacts/data/fastumi_original_sample/derived_command20/README.md) 与 [实际MuJoCo结果](../artifacts/data/fastumi_original_sample/derived_command20/mujoco_coupled250/README.md)；此新例尚无同例PhysX或视觉闭环证据。

固定世界TCP位姿的真实渲染图像闭环见 [diagnostic_pose_visual_control/README.md](../artifacts/runs/diagnostic_pose_visual_control/README.md) 和 `closed_loop20/summary.json`：复用权重1控制组bundle，经30 Hz腕部RGB-D、ArUco／深度测量、10 ms延迟队列驱动50 Hz单18关节策略，MuJoCo完整 **20秒／1000控制步**，退出码0、墙钟 **27.38秒**。固定marker到目标的变换在运行前冻结；仿真里程计提供世界相机定位，目标真值只用于场景和评分，控制器消费图像测量。

601次采集均成功测量，600个独立测量送达控制（终点帧晚于末次控制）；**999/1000步**输入有效，唯一保持发生在首步等待延迟图像，最大使用测量年龄 **40 ms**。此后无漏检、失跟或保持，采样观测和qpos/qvel全部有限，现有跌倒检测未触发。全程TCP位置／朝向RMSE为 **.018896 m／.085253 rad**，2秒后 **.018855 m／.085240 rad**，终点 **.023087 m／.061185 rad**；图像推导目标的测量RMSE为 **.038886 m／.066913 rad**。固定marker到TCP偏移可能放大朝向误差的位置贡献，本次未隔离原因。

该固定目标场景没有注入遮挡、噪声或深度缺失；最低采样基座高 **.246403 m**，未记录非足接触和逐子步力矩饱和，不能据无跌倒宣称稳定站立。结果属于临时参数和名义D435标定下的图像→测量→控制工程证据，尚不覆盖移动目标、遮挡恢复、真机精度或60秒视觉验收。README包含复现入口，`closed_loop20/wrist-view.mp4` 是20秒实际腕部输入视角。

首次 PhysX 后测退出码 **139**，`eval_post1000_console.log` 保留 OpenBLAS shutdown／fork 原生崩溃；仅设置 `OPENBLAS_NUM_THREADS=1` 后在 `eval_post1000_openblas1/` 重试，operator 确认退出码 **0**。失败记录不替换为成功，实际前后比较只消费重试报告。

同一最终策略的独立 MuJoCo 后测 `eval_mujoco_post1000/report.json` 已完成，operator 确认退出码 **0**：8例各20秒、零跌倒，平均位置 RMSE **.052304 m**、朝向 **.213909 rad**。跨引擎比较 `cross_engine_post1000.json` 退出码 **0**；PhysX／MuJoCo 平均位置分别 **.040647／.052304 m**、朝向 **.219503／.213909 rad**，现有仅位置 tracking pass rate 为 **6/8／5/8**。8例都是动态轨迹，静态到达率降幅及其是否达标均为 `null`；`eligible_for_acceptance=false`、`pose_acceptance_passed=null`。该工程结果保留 `trained=false`，临时参数、单种子和合成 FK 引用不能证明硬件有效性或完成正式里程碑。

独立 MuJoCo 入口已支持明确的诊断／临时参数模式。真实几何与本轮初始策略的 CPU 构造保持仿真时间为零，验证了编译前被动参数、力矩、初始高度和派生惯性常量，未加载 Isaac／RSL-RL／tensordict。相关 runtime 检查 27 passed、1 skipped（既有 Isaac 测试需要 tensordict）；报告接口测试替换了步进，不能证明闭环动力学。该准备阶段的检查不替代上文现已完成的真实 MuJoCo test8 闭环结果。纯动态套件的跨引擎比较已修复缺少静态到达率时的计算错误，缺失指标保留为 `null`。

KMA dogfood 持续以实际研究循环效率为目标，而非安装、角色数量或检查通过数。此次删除 bundle 内无 runtime 消费用途的重复 metadata，完整记录仍留在 checkpoint／run；同一初始化输入 manifest 从 23,820 降至 15,056 字节，减少 8,764 字节，见 `bundle_metadata_size_comparison.json`。批量评估不再受旧 exact-hold 精确重放条件阻塞，并删除新增的完整 bundle hash 比较门槛；已有加载时文件完整性／资产检查仍执行。上面的批量前测是实际结果，旧 exact-hold `passed=false` 原样保留。这些具体修正支持继续试用，不证明 KMA 在没有人工提示时一定能自然作出有效取舍。

## 已完成的历史检查

| 检查 | 对象 | 结果 |
|---|---|---|
| 训练环境启动 | Isaac Sim 6.0.1 + Isaac Lab `v3.0.0-beta2.patch1` | 实际 PhysX 启动及步进通过 |
| 依赖一致性 | 三个 Conda 环境 | `pip check` 均通过；运行环境无 Isaac Lab/RSL-RL |
| SDK MDH / URDF | Piper-H，100 个随机关节姿态 | 位置与旋转矩阵差均小于 `1e-6` |
| URDF / MuJoCo FK | 组合模型，50 个随机基座及关节姿态 | 足端与 TCP 位置/旋转矩阵差小于 `1e-8` |
| URDF / PhysX FK | 含 D435 的组合参考模型，25 个随机姿态 | 最大位置差 `6.03e-7 m`，姿态差 `8.10e-5°` |
| 质量与静态重力 | 同一组合参考模型 | 总质量约 25.682 kg；最大关节重力力矩差 `6.36e-6 Nm` |
| 接触传感器映射 | 组合参考模型 | 各传感器映射至对应名称的单个刚体；未完成站立、沉降与摩擦性能验收 |
| PPO + 辅助学习 | 合成 18 关节盒体机器人 | 4 环境、16 步 rollout、2 epoch，含随机化、部分重置、有限损失及策略导出 |
| 策略迁移 | 合成机器人同一导出网络 | GPU/CPU 最大动作差 `2.98e-8`，小于 `1e-5`；独立 MuJoCo 步进 200 次 |
| 固定基座阶跃 | 合成机器人，18 个单关节案例 | 每例 0.5 秒，通过 `0.002 rad + 位移10%` 容差；最大差约 0.002491 rad |
| 真实 RGB-D 渲染 | 独立合成标记/虚拟相机 | 三维位置误差约 0.076 mm |
| 视觉控制链路 | 合成机器人、虚拟腕部相机 | 30/50 Hz 调度、延迟、深度缺失、噪声、遮挡保持及恢复通过；极短测试测量 RMSE 约 1.68 mm |
| 固定测试输入 | 旧100个position候选回合 | 已生成并校验哈希，含0.3–2 m到达和60秒轨迹，尚未实跑；100例均缺朝向，当前pose `load_suite`拒绝，不能作为已有100例位姿套件；精确可达性审核待完成 |
| 消融准备 | 三种训练种子 | 已生成 48 个课程/预测/采样配置作业；尚未运行；原FastUMI一份本地episode已检查但尚未正式TCP转换／导入，不声称具备示范消融结果 |

短时视觉测试仅 0.14 秒，观测到遮挡结束后约 40 ms 恢复；这个时长只能检查代码路径。它既不能证明稳定性，也不能外推实际 D435 的定位精度。夹爪控制、真机定位与抓取接触不在本版范围内。

小型 JSON 证据保存在 `docs/reports/`；完整轨迹、日志、网格和测试权重在 `artifacts/` 或 `assets/generated/`。单元测试覆盖因果隔离、命名映射、PD 延迟/饱和、历史观测、数据变换、分割、相机投影和测试集防篡改。最终测试计数见 `docs/reports/checks.json`。

## 真实组合几何 PPO 工程诊断 · 2026-09-09

`artifacts/runs/diagnostic_geometry_seed0/` 保存完整日志、`run.json`、`metrics.jsonl`、检查点和 `trained=false` 包。实际加载资产 `1fe12b296656e1684165d60cdfb36e5c003cbb777f4d0e8525d14516d4f9176a` 的 `usd/robot_3/robot.usda`，保留 canonical/USD 校验，未重建几何。显式临时参数见 `configs/diagnostic_actuators.json`，不改变 M0 或原性能验收条件。

4 环境 × 16 控制步（每环境 0.32 秒），seed 0、1 次迭代、2 epoch × 2 minibatch，实测完成 4 次优化器调用，Actor 参数最大变化 `0.00160790`。所有观测、动作、奖励、每个物理子步的状态/力矩、优化梯度/参数和损失有限。value loss `0.390444`，surrogate `-0.00196074`，entropy `13.0535`，KL `0.251868`，辅助损失为零；吞吐约 `71.79` 环境控制步/秒，包含本次诊断检查与优化，不代表大规模训练吞吐。

TCP 误差均值 `0.240611 m`、最大 `0.619186 m`、最后一步环境均值 `0.567925 m`。11,520 个关节物理子步样本中力矩饱和为零；采样动作最大绝对值 `1.79678`，控制器按既有约定裁剪到 ±1。奖励范围 `[-0.125680, -0.00381133]`。跌倒/重置均为零，但初始足端 link 原点高约 `0.211379 m`（不是足底间隙），基座由 `0.5 m` 降至最低 `0.261892 m`：短时结果反映明显沉降，不能证明稳定站立或无跌倒风险。下一项具体问题是初始站姿与沉降行为。

首次执行后发现嵌套 contact 路径警告，保留在 `attempt1/`。复用已有传感器映射断言重跑相同短实验，30 个传感器均对应其指定单个刚体，物理/PPO 数值与首次相同；警告仍保留在日志，未改变接触实现。独立 runtime 环境成功校验并加载导出包及完整临时参数/身份；正式加载拒绝 `trained=false`。相关输入门控/哈希/关节限位/检查点兼容测试和现有 CPU PPO 测试共 4 项通过。

这是实际几何上的有限 PPO 集成证据，尚非训练完成、硬件有效性或任务性能证据；未进行独立 MuJoCo 闭环验证。

## 紧凑重置候选与初始高度 · 2026-09-09

用户指出原渲染姿态高举手臂，不应直接视为工作初态。`artifacts/preview/initial_pose_candidates/` 对原姿态和三个折叠候选进行相同角度/尺度渲染、URDF 限位和 MuJoCo 静态接触检查。其中曾用臂 `[0,0.1,-0.15,0,0.2,0]` 作对照（现已由下述 SDK 六零参考替代）：肩/肘为配置的 ±0.1 rad 动作保留范围，臂+相机最高点相对基座从约 0.653 m 降至 0.431 m，TCP 前伸从 0.356 m 降至 0.212 m。四个姿态均未出现非相邻碰撞模型接触；保留 canonical 相邻排除，不把视觉或静态检查当作硬件安全证明。该姿态仅为紧凑仿真重置候选，不代表厂商 home、标定零位、断电姿态或稳定工作初态。

高度测量复用真实 28 mm 足部碰撞球：基座 0.5 m 时球底间隙 0.183378493 m；基座取 `0.317621507 m` 时四足初始间隙为 1 mm。缺省正式配置仍为 0.5 m。`diagnostic_settling_seed0/` 保留旧臂姿态高度对照；已启动的 `diagnostic_geometry_grounded_seed0/` 也是**旧显示臂姿态**，不代表新候选。

`artifacts/runs/diagnostic_settling_compact_seed0/` 为新候选在两个高度各 2 秒零动作 MuJoCo 对照。高度从 0.5 m 改为测量值后，足部峰值法向力约 528→117 N，末段最大关节速度 0.0743→0.0724 rad/s；两次均有限、无引擎警告、无跌倒阈值触发，最终四足接触。较低起点最终基座 0.2553 m、俯仰约 -6.29°，末半秒仍变化约 7.15 mm；所以减小落地冲击不等于稳定站立。肩/肘随后靠到软限位，也未精确保持指定 q0。

`artifacts/runs/diagnostic_geometry_compact_seed0/` 完成相同 4×16 步、2×2 minibatch 的实际 PhysX PPO。实测初始基座 0.317621499 m、足部 link 中心约 0.029 m，全部 4 次优化器更新完成，Actor 最大参数变化 0.00160786；观测、物理状态/力矩、动作、奖励、梯度/参数及损失有限。TCP 误差均值 0.13236 m、最大 0.20827 m；无跌倒/重置、11,520 个关节物理子步零力矩饱和；约 72.11 环境步/秒。`trained=false` 包已在 runtime 独立校验。此短实验验证配置实际生效及有限优化，不证明学习效果，且不能把改变初始状态后的 TCP 数值直接归因为策略改善。

`artifacts/runs/diagnostic_arm_hold_seed0/` 进一步固定基座，只对新候选切换重力开/关，各 2 秒，保持原 PD、被动参数和力矩界限。关重力时精确保持 q0；开重力时肩/肘靠软限位，且没有非相邻臂/相机接触。初始肩/肘重力偏置约 +2.220/-5.909 Nm，而零误差 PD 初始输出为零：当前首先暴露的是重力负载下的姿态保持问题，未进行增益调整、重力补偿或力矩扩界。旧显示姿态的存量轨迹中，末段高速主要来自 arm6/arm4 且每个 2 ms 采样都反向，符合离散振荡表现，但其具体原因没有被该新姿态对照单独证明。目标工作初态仍需验证犬体稳定站立与紧凑手臂受控/受支撑保持；不能用电机断电的自由下垂替代。

## 当前诊断默认：SDK 六零参考 · 2026-09-09

臂 q0 改为 `[0,0,0,0,0,0]`，来源为固定版本 [Piper-H SDK demo](https://github.com/agilexrobotics/pyAgxArm/blob/e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b/pyAgxArm/demos/piper_h/test1.py#L104) 的六零 `move_j`。URDF 已包含 MDH 固定框架偏置，不另加零点偏移。它是有来源的紧凑参考，不声称 GUI/固件厂家 home 数值已验证。控制始终启用；`render_asset.py` 从诊断参数读取相同 q0，当前 `artifacts/preview/robot.png` 显示该参考。历史候选和实验全部保留。腿部姿态与 0.317621507 m 诊断初始高度不变，缺省正式高度仍为 0.5 m。

`artifacts/runs/diagnostic_settling_sdk_zero_seed0/` 的 2 秒零动作 MuJoCo 实验暴露不稳定保持：末段 arm4/arm6 最大速度约 20.14/25.32 rad/s，最大关节误差 2.158 rad，软限位最大越界约 0.0631 rad，力矩饱和 9.92%。最终基座约 0.2327 m、俯仰 -11.38°。状态有限、引擎无警告、未发现非相邻臂/相机接触，四足最终接触且没有触发跌倒阈值，**但不满足稳定站立/紧凑保持的工作初态**。实际 q/qd、足接触、力矩与完整轨迹已保存。下一具体问题为该参考与临时被动动力学/离散 PD 下的腕关节振荡和限位负载；未盲目调整增益、力矩或加入补偿。

同一参考的 `artifacts/runs/diagnostic_geometry_sdk_zero_seed0/` 实际 PhysX 4×16 PPO 仍完成 4 次优化器更新，Actor 最大变化 0.00160760，全部有限；实际初始基座 0.317621499 m，力矩饱和 171/11,520（1.484%），无跌倒/重置，TCP 误差均值 0.09938 m，约 71.24 环境步/秒。runtime 已校验 `trained=false` 包及参数/身份一致。0.32 秒有限 PPO 集成不能覆盖 2 秒零动作中暴露的保持问题，更不能代替稳定工作初态或原性能验收。

`artifacts/runs/diagnostic_sdk_zero_wrist_numerics/` 做了一组 0.2 秒数值判别：固定基座、关重力、无接触，同一六零目标，只给 arm4/arm6 初始 ±0.0001 rad 扰动。原 2 ms 显式 PD 下误差放大到约 2.024/2.224 rad，峰值速度约 94.7/108.4 rad/s；保持相同参数、将物理/反馈步长临时缩至按质量矩阵预先计算的 0.0971 ms 后，误差衰减至约 8.2e-6 rad、无饱和。零臂 armature 假设下，质量矩阵最小特征值约 0.0001553 kg·m²，模式主要为 arm4/arm6 反向运动；`λmax(M^-1 D)≈5151/s` 的线性显式阻尼步长界约 0.388 ms，小于 2 ms。这隔离出即使无重力/碰撞仍存在的显式 PD 与临时惯性/离散步长问题；不能把全部失稳归给重力。小步长仅用于判别，产品 2 ms/50 Hz、增益、力矩和 armature 均未修改。应先核实有来源的电机反射惯量/执行器模型，再决定实现；不把零臂 armature 或该数值界限当作硬件结论。

## 当前臂 armature 假设及加载检查 · 2026-09-09

诊断默认六臂 armature 从零改为 0.005 kg·m²，来源为 [标准 Piper 厂商仿真](https://github.com/agilexrobotics/agx_arm_sim/blob/f8cd8b147c75d59e14f90fb0646770eefa268ed0/mujoco/agilex_arm/agilex_piper/piper.xml#L4)。这是 **standard Piper vendor-simulation value provisionally transferred to Piper-H**，不是 Piper-H 实测转子/反射惯量。未迁移该文件的 frictionloss 0.3，未改变臂 damping/frictionloss、增益、力矩、q0、50 Hz/2 ms 合同。

`artifacts/runs/diagnostic_sdk_zero_armature_probe/` 先只变此参数重复原无重力/接触的 0.2 秒腕扰动：实际按名加载 .005，2 ms 下误差衰减至约 ±7.25e-6 rad、无饱和；阻尼线性步长界约 12.888 ms。固定基座开重力后腕振荡消失，但 J5 有约 0.131 rad 负载偏移、J2/J3 仍受软限位支撑。

`diagnostic_sdk_zero_armature_settling10s/` 延长全身零动作保持到 10 秒：最后两秒最大关节速度 0.00491 rad/s、基座高度变化 0.556 mm、俯仰变化 0.126°，四足持续接触，90,000 个关节物理样本无饱和/跌倒，状态有限。最终基座却仅 0.22851 m、俯仰 -14.16°，软限位最大越界 0.00205 rad。保存终态的几何重放还发现两个后大腿 `collision_2` 与地面接触（约 22 μm 浅穿透，重放不声明接触力）。这是趋于安静的低姿态，不是已证明的稳定站立。原控制台样本数继承了 2 秒脚本常量；`result.json` 已据 5,000 步轨迹纠正为 90,000，原执行脚本和日志保留。

`diagnostic_physx_sdk_zero_armature_hold2s/` 通过现有 WholeBodyEnv 运行 2 秒零动作：实际六臂 armature 为 .005，最终基座 0.28106 m、俯仰 -1.12°，末半秒最大关节速度 0.00741 rad/s，四足有力、零饱和/跌倒且全部有限。对应 MuJoCo 2 秒为 0.25708 m/-5.95°；PhysX 还曾记录 `arm_flange_link`/`rgbd_stand` 大于 1 N 的接触力，MuJoCo 臂/相机接触列表为空。未辨认该 PhysX 接触对，不声称双引擎加载行为一致；这是当前具体未解问题。

`diagnostic_geometry_sdk_zero_armature_seed0/` 完成现有 4×16 PhysX PPO：4 次优化器更新、Actor 最大变化 0.00160793、有限损失/状态/梯度，零饱和/跌倒；约 68.85 环境步/秒，TCP 误差均值 0.09842 m。`run.json` 保存 PhysX 按名直接读取的六臂 .005、源代码/配置/资产身份；runtime 已校验导出包和相同 metadata，仍为 `trained=false`。这些工程结果既不闭合硬件参数，也不替代原任务性能验收。

## 当前 USD：固定连接组碰撞过滤 · 2026-09-09

报告过滤先确认 `arm_flange_link ↔ rgbd_stand` 内部接触力峰值约 45,014 N；单对运行时排除使两端力归零，而最大基座坐标差仅 1.16 μm。这证明内部碰撞伪影存在，**但它不能解释当前两引擎的明显站姿分歧**。历史接触/碰撞奖励/Critic 接触观测受此影响，不因有限 PPO 更新而升级为有效物理行为证据。

持久修正在 USD producer：`RobotTree.fixed_collision_pairs()` 只枚举由固定关节连通的 collision-bearing link 对，可穿过无质量固定 frame，绝不跨 revolute/continuous/prismatic。`convert_usd.py` 唯一解析对应刚体，保留已有关系并添加 `FilteredPairsAPI`；缺失/重名会失败，不合并刚体、不改惯性、frame、sensor 或 MJCF。新入口为 `usd/fixed_groups_20260909/robot/robot.usda`，26 对及 producer 哈希已记录；旧 `robot_3/robot.usda` 等历史 USD 文件校验保持不变。26 对是拓扑规则推导，并不声称每一对都观察到碰撞。

`artifacts/runs/diagnostic_physx_persistent_fixed_groups/` 独立加载新旧 USD：30 个刚体名称/路径、29 个 joint frame、质量/惯性完全一致，world transform 最大差为零，26 个关系确已落盘。随后同一 2 秒零动作报告检查不再修改运行时物理：法兰↔支架过滤力及 net force 均为零，其他非足端没有超过 1 N，四足最终约 50/50/77/76 N。原生按名读回的腿 friction 三列为 `[1,1,.2]`、`[1,1,.2]`、`[2,2,.5]` 重复四次，臂为零；native stiffness/damping 为零，max force/velocity 为 `1e9/1e6`，与既有显式 PD 路径预期一致。此读回消除了加载参数不明的问题，不能证明双引擎被动响应相等。

`diagnostic_geometry_fixed_groups_seed0/` 完成现有 4×16 PPO：4 次更新、Actor 最大变化 0.00160773，状态/损失/梯度全部有限，零饱和/跌倒；约 71.41 环境步/秒。runtime 包校验通过，`trained=false`，run metadata 与新 USD locator、26 个关系及当前 producer/source 哈希一致。4 项固定拓扑边界测试通过。站立尺度 `.2` 的常量动作候选继续只保存在 `diagnostic_standing_authority_seed0/`，未改变默认 `.1/.005` 诊断配置或原性能验收。

## MuJoCo 实验初始化纠正及重验 · 2026-09-09

`diagnostic_real_leg_pd/` 的真实腿阶跃暴露了 **artifact 脚本在编译后改 armature，却未刷新 `dof_invweight0`** 的问题；friction constraint 实际已启用，不能把旧差异归因于“MuJoCo 未启用摩擦”。同参数写入 XML 后编译与 `mj_setConst` 对照完全一致。生产 MJCF 路径在编译前写入参数，无需添加 runtime 刷新循环。原始失败及旧结果均保留；此前 `diagnostic_sdk_zero_armature_settling10s/`、`diagnostic_standing_authority_seed0/` 及 `.005` 系列较短重力/负载 artifact 的加载接触数值不能继续作为正确常量下的证据。以下重验**替代相应旧证据**，不静默改写历史；静态重力/支撑计算不受此次模型常量问题影响。

`artifacts/runs/diagnostic_mujoco_constants_revalidation/` 保存编译输入 XML、实际按名 `dof_invweight0`、精确 spec/action、2 ms 状态/力矩/饱和轨迹，以及每步全部接触 wrench 和关节限位反力。仅纠正参数编译时机，无增益、几何、默认 spec 或动作调参。默认 `.1` 零动作 10 秒终态为 z **0.228777 m**、pitch **-13.355°**；末两秒最大关节速度 **0.02122 rad/s**、高度范围 **1.413 mm**，四足持续有力，但 RR/RL 大腿分别从 **7.868/7.942 s** 接地，不能称为稳定站立。精确重放旧 `.2` 腿尺度及保存的常量动作候选，终态 z **0.316169 m**、pitch **+0.304°**，末两秒最大速度 **0.001868 rad/s**、高度范围 **0.0226 mm**，全程仅四足接地。两例各 90,000 关节物理样本均有限、零饱和、未触发既有跌倒阈值；J2/J3 限位反力持续存在，最大软限位越界分别 **0.001450/0.001272 rad**，候选仍不是主动臂策略或硬件验收。

与持久过滤后的既有 PhysX 零动作记录在 **0.02–2.00 s、100 个 50 Hz 样本**重叠比较，默认 spec 与初始高度一致：MuJoCo 2 秒 z **0.273441 m**，PhysX **0.281055 m**，最大高度差 **7.615 mm**、最大关节差 **0.06754 rad**，仍不等价。保存的 PhysX 记录没有姿态角，不重构 pitch 差；其足力为 net magnitude，MuJoCo 为 normal force。`.2` 候选与该 PhysX 零动作的对照仅作不同输入描述，不能当作引擎一致性检验。`overlap_comparison.json` 保存逐项误差与证据边界。

同目录 `wrist/` 用正确编译常量重放 `.005`、固定基座、无重力的 0.2 秒 J4/J6 ±0.0001 rad 扰动：100 个 2 ms 步后分别为 **-7.175e-6 / +7.251e-6 rad**，无接触、零饱和、状态有限；线性阻尼步长界仍为 **12.888 ms**。这替代旧定量衰减证据，并不将 standard Piper 仿真 armature 升格为 Piper-H 实测值。修正后的无重力真实腿阶跃最大位置差为 hip **0.003172 rad**、thigh **0.004646 rad**、calf **0.011418 rad**；前两者满足既有软件参考，calf 不满足，剩余被动响应差异的具体求解器原因未闭合。全部重验只属工程证据，原任务性能验收不变。

## 主动臂展开与策略初始化候选 · 2026-09-09

`artifacts/runs/diagnostic_active_arm_unfold/` 在正确编译的 MuJoCo 模型及当前 PhysX USD 上验证了主动展开。候选仅调整动作尺度（腿 `.2`、臂覆盖配置范围），保持现有增益、力矩、被动参数、50 Hz/2 ms 合同；使用离线静态重力计算的 PD 目标，不增加在线补偿控制器。`plan.json` 保存精确动作，默认诊断配置未改变。

从重置直接输出 `actions[1]` 常量动作、没有预热阶段的 3 秒对照中，MuJoCo 终态 z **0.316291 m**、pitch **0.136383°**、J2/J3 **0.100975/-0.149681 rad**；PhysX 为 **0.313401 m**、**0.080698°**、**0.100593/-0.150002 rad**。两者状态有限、零饱和、未触发跌倒阈值。MuJoCo 全程仅足地接触且没有关节限位约束；PhysX 四足承载、没有非足部超过 1 N 的净接触力，未读取原生关节限位反力。末半秒最大关节速度分别为 **0.002720/0.003397 rad/s**。PhysX 仍有约 **0.369 mm** 的末半秒高度变化，故这支持短时主动展开和常量策略初始化，不能外推长期站立、学习效果或硬件有效性。

`artifacts/runs/diagnostic_near_goal_learning_seed0/` 已据此构造与既有 `--initialize-from` 严格兼容的初始化：只将 Actor 最后线性层权重置零并写入精确常量均值，保留隐藏层及全部参数可训练。CPU eager、严格加载和 JIT 输出最大差为零，runtime 校验通过，包仍为 `trained=false`。冻结的 16 个近距离目标用于后续前后对照；此初始化检查不属于已经完成的任务学习。

## 近距离 PPO 学习前后对照 · 2026-09-09

`artifacts/runs/diagnostic_near_goal_learning_seed0/README.md` 的 **Executed result** 及 `paired20.json`、`paired20_summary.json` 保存已完成的真实几何学习结果。`train20/` 完成 32 环境 × 256 步 × 20 次迭代，共 163,840 transitions、80 次优化器更新；全部有限、零跌倒、零力矩饱和，runtime 独立加载导出包并确认 `trained=false`。冻结输入为 12 个静态近距离目标和 4 条线轨迹；有效顺序前测为 `eval_pre_retry/`，后测为 `eval_post20/`，没有排除较差案例。

两次评估各 16 例均完成完整 20 秒，全部有限，无跌倒、动作裁剪或所有物理子步力矩饱和；50 Hz 观测未发现非足部净接触力超过 5 N，这不是接触对或每个物理子步的无接触证明。2 秒后的平均逐例 RMSE 为 **0.0630698 → 0.0624163 m**，但 **5 例改善、11 例变差，四条线轨迹全部变差**；连续至少 2 秒处于 5 cm 内的案例 **9 → 8**。这是可用的负面结果：总体均值略有改善，未显示有用的跟踪学习。相同观测下改变目标，最大动作变化从零增至 **0.00788713**，仅证明目标响应非零，不能证明策略正确跟踪目标。没有硬件有效性或 M1–M4 验收结论。

首次前测 `eval_pre/` 在第二次 reset 失败：`torch.inference_mode()` 内创建的 PD target 在上下文外被原位修改。artifact 评估器改用 `torch.no_grad()` 后，`eval_pre_retry/` 完成 16 例并精确复现第一例；`evaluate_before_no_grad.py` 和原日志保留失败原因，未改生产控制器或策略数值。另仅纠正报告中的完整 canonical bundle hash，原字段保存在 `eval_pre_retry/results_before_bundle_hash_correction.json`，不重写历史。

## 批量评估候选未通过 · 2026-09-09

`artifacts/runs/diagnostic_batched_eval_probe/initial_two_comparison.json` 对初始化策略的首两例进行顺序/批量比较，身份、输入、动作、完整时长及预先固定的状态/误差容差通过，但两例的 `exact_continuous_5cm_hold_s` 均失败，整体 **`passed=false`**。虽然是否保持至少 2 秒的布尔结果一致，不能代替精确 hold 时长要求。当时未放宽标准、未进行旧 16 例批量评估，旧学习结论来自完整顺序前后测。该精确重放失败不作为当前位姿批量评估的前置条件，也不由新结果改写。

## 4096 环境候选与有限地面修复 · 2026-09-09

原 `create_scene` 使用 200×200 m cuboid 地面，8 m 间距的实际 cloner 网格在 1024/4096 环境时延伸到 ±124/±252 m，分别有 348/3420 个原点位于旧地面外。修复只按 `8 * (ceil(sqrt(num_envs)) - 1) + 200` 设置宽度，保留原点外 100 m 余量，不改变材质、厚度、高度、prim path、机器人或环境间距。`artifacts/runs/diagnostic_4096_scale/cpu_coverage.json` 执行安装版本 cloner 函数并使用 canonical URDF FK 的实际足部球形碰撞几何：1/32/1024/4096 环境宽度为 200/240/448/704 m，重置足部最小边缘余量约 **99.753941 m**。这只验证 CPU 布局和重置几何；下述真实容量运行独立提供步进和资源证据，不能据此重构未保存的 runtime XY 原点或动态地面覆盖。

候选配置和固定上游来源见 `artifacts/runs/diagnostic_4096_scale/README.md`，历史续训命令见 `docs/runbook.md`。相比已完成 train20 配置，只改为 4096 环境、24 步 rollout、5 epoch、4 minibatch；复用原 `candidate_spec.json` 的全部参数和来源，保持单个 18 关节 Actor、任务、奖励、2 ms/50 Hz、stage 0、初始学习率 `.001` 和 adaptive KL `.01` 等选项。

`capacity3_summary.json` 记录真实 **4096×24×3** 容量运行正常退出，共 **294,912 transitions、60 次优化器更新**，全部有限，零跌倒，53,084,160 个子步关节样本零力矩饱和。实测约 **29.8k–32.3k 环境控制步/秒**；每 500 ms 采样的**设备级**显存峰值为 **6,652 MiB**，包含后台占用且可能漏掉瞬时峰值，不是精确进程峰值。每环境只经历 **1.44 秒**，这是容量证据，不是长回合或跟踪验收。runtime 独立加载导出包通过，仍为 `trained=false`。

首三轮 KL 为 **42.72497、0.06927、0.03585**，保存的 optimizer LR 已到 **`1e-5`**。`kl_probe/README.md` 与 `result.json` 的 CPU 固定观测实验核实 Actor/Critic/std/normalizer 精确加载；原 train20 LR 约 `5.85e-5`，`--initialize-from` 按现有语义重建 fresh Adam 并从 `.001` 开始。窄 std 下 fresh Adam／高 LR 的单步敏感性明显，normalizer 变化也贡献 KL，支持低 LR 严格恢复进行有界观察。固定观测与合成优势不是首轮真实 rollout，不能把该判别当成精确因果重放；42.725 是逐 minibatch、18 动作维求和 KL 的均值，不是最终策略 KL。

后续首次 `train50` 在追加 **0 次**迭代时失败，日志 `train50_console.log` 保留 `RNG state must be a torch.ByteTensor`：checkpoint 的 `map_location` 将 CUDA RNG ByteTensor 搬至 GPU，而恢复接口要求 CPU。最小生产修复 **`f114ae5`** 仅将 RNG 张量转回 CPU；5 项回归测试（含实际 CUDA）、真实 capacity checkpoint 的 CPU/CUDA 随机序列精确恢复及独立 review 均通过，未改变 Actor/Adam 的设备、学习率或配置。证据见 `kl_probe/resume_test.log`、`kl_probe/resume_actual_checkpoint.json`。

修复后的 **`train100/` 已完成 97 个追加迭代**；与 capacity 合计 100 次、9,830,400 transitions、2,000 次优化器更新。它继承 Adam 与 `1e-5` 学习率，仿真回合重新开始。续训全部有限，零跌倒/饱和、8,192 次超时重置；最后十轮 KL 为 `.00844–.01579`，最终学习率自行恢复至约 `.000256289`。设备显存采样峰值 6,980 MiB；`checkpoint_000099.pt` 和 bundle 在 runtime 独立校验通过。原冻结 **16×20 秒顺序 `eval_post100/` 已完成**，完整配对结果如下。

`artifacts/runs/diagnostic_4096_scale/paired100_summary.json` 比较初始化、train20 与 train100，2 秒后的平均逐例 RMSE 为 **6.307 → 6.242 → 6.844 cm**。train100 相对初始化 **6 例改善、10 例变差**，相对 train20 **5 例改善、11 例变差**；四条线轨迹均退步，其平均 RMSE 为 **2.722 → 3.199 → 4.653 cm**。train100 的连续至少 2 秒处于 5 cm 内案例为 **9/16**（初始化 9/16、train20 8/16）。16 例全部完整运行 20 秒，无跌倒、50 Hz 观测的非足部净接触力超过 5 N、动作裁剪或所有子步力矩饱和；净接触观测仍不能证明每个物理子步无接触。相同观测下改变目标的最大动作变化增至 **0.0311694**，但任务指标整体退步，不能宣布学会跟踪。所有结果仍为 `trained=false` 的工程证据，无硬件有效性或原里程碑验收结论。

## 旧位置任务 balanced-y 完整结果与新目标边界 · 2026-09-09

`artifacts/runs/diagnostic_balanced_axis_learning/full16_summary.json` 记录平衡静态 y±0.05 m 训练后的原固定 16 例完整顺序评估。2 秒后平均逐例 RMSE 为 **5.466 cm**，相对初始化 **6.307 cm** 有 **9 好 7 差**，相对 train100 **6.844 cm** 有 **13 好 3 差**；连续至少 2 秒处于 5 cm 内为 **10/16**。四条线的均值为 **3.251 cm**，虽优于 train100 的 **4.653 cm**，但仍差于初始化的 **2.722 cm**，其中 **3/4** 相对初始化退步。16 例均完整 20 秒，无跌倒、50 Hz 非足部净力超过 5 N、动作裁剪或所有子步饱和。该结果显示旧位置任务的部分改善，不能证明完整轨迹跟踪，更不能验证未训练的末端朝向。

六轴位置候选已取消且未启动。当前末端位姿接口、1000轮学习及双引擎后测见本文开头；位置改善而朝向全退步，完整位姿成功尚未成立，新增朝向验收阈值待确定。单个 18 关节 Actor 保持不变，不启动独立 AS2／Piper-H 预训练；随机教师 CPU 原型仅备用，不具有预训练或技能迁移证据。原日志、指标与验收数值保留，不升级为新任务或硬件有效性证明。

## 尚未完成的验收

- Piper-H 质量/限位/驱动版本，转接板/支撑/线缆，真实相机惯性及标定等 M0 参数。
- 精确硬件执行器阶跃、沉降、足端摩擦和滑移测试。
- 末端位姿策略的完整任务有效性与性能验收；1000轮工程学习显示位置改善、朝向退步，随后权重对照改善朝向但损失PhysX位置精度，新增朝向验收阈值尚未确定。
- 1024/2048 环境吞吐测试、完整训练检查点、三个训练种子的任务性能与消融结果。
- 双引擎各至少 100 回合的闭环性能对照、完整 60 秒视觉跟踪与演示视频。

不得用诊断资产、合成测试策略或上述短时测试替代这些验收。
