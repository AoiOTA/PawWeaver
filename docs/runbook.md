# 运行手册

从 `/home/lyb/pawweaver` 运行。若 shell 继承 ROS 的 `PYTHONPATH`，在 Python 命令前加 `env -u PYTHONPATH PYTHONNOUSERSITE=1`；测试脚本已处理这一点。

## 当前目标与运行边界

**全身能力目标**：仅有18维统一动作接口不等于完成WBC。当前任务须覆盖低位、高位、侧向、远距离及朝向变化的世界系末端目标，使腿部支撑、身体升降／倾斜与机械臂按任务需要共同协调。主动下蹲、倾身和支撑转移属于原目标的未完成部分，不能降为以后可选扩展；仍不输入底盘姿态／速度命令或手写步态。现有UMI-inspired实验偏水平、小朝向变化，其结果只证明实际覆盖范围。后续修改目标分布和姿态限制应保留原性能验收，并依据模型净空和接触区分正常低姿态与跌倒。

**当前运行状态**：[广范围6D训练](../artifacts/runs/diagnostic_pose_learning/wbc_random_training/README.md) 的2000轮和双引擎十项后测均已实际退出0，最后一项MuJoCo静态16例于2026-09-10 05:06:32 UTC结束。训练196,608,000 transitions／40,000次更新、7538.87秒，全部数值有限，累计22,938次跌倒／76,267次重置。PhysX／MuJoCo独立64例分别11／12例通过既有位置跟踪条件、24／36例跌倒；静态16例分别3／2例达到既有位置到达条件、13／14例跌倒。实际流程完成不等于任务完成，尚未达到全身控制目标；朝向正式验收阈值未指定，`trained=false`。逐例与原始报告索引见 [evaluation_summary.json](../artifacts/runs/diagnostic_pose_learning/wbc_random_training/evaluation_summary.json)，下文保留历史实验。

**历史UMI-inspired实验**：[UMI-inspired 训练方案](../artifacts/runs/diagnostic_pose_learning/umi_recipe/README.md) 的1000轮和双引擎test8／far60四项后测已全部实际退出0，仿真资源已释放。训练98,304,000 transitions／20,000次更新、3654.53秒，全部数值有限，累计13,865次跌倒。两引擎local4均完整20秒，位置／朝向RMSE均值分别为PhysX .006117m／.010725rad、MuJoCo .003630m／.010436rad；moving均3/4中途跌倒，far分别6.68秒／1.64秒跌倒。局部精度改善，移动与远距稳定性不支持任务成功，不采用为完整WBC成果。主比较文件已按逐例共同时间窗计算；MuJoCo far没有2秒后的共同窗。下文保留历史实验；本轮为整套方案试验，不作单变量归因。

**当前推进方向**：[全身工作空间目标](../artifacts/runs/diagnostic_pose_learning/wbc_workspace/README.md) 与256条训练／64条独立采样广范围6D轨迹已用于上述实际训练和测试。历史前测64例fresh PhysX均完整60秒无跌倒，位置／朝向RMSE均值.749939m／2.038435rad，保留作为前后比较背景。`7945f9a`接通的`demonstrations_only`、可配置最低高度和双引擎同帧跌倒原因记录已被此次实际运行使用。当前实验仅采样新训练文件库；依据base=.18m四足支撑静态几何见证，使用最低基座高度.15m、原base-up-Z .35及body_tilt奖励0，默认高度.2m的历史实验仍保留。这些设置不把低姿态自动判成稳定。保存轨迹已显示臂、身体和腿共同参与任务，但存在长期高抬单足、三足承载及倾倒，不能视为稳定广泛WBC。[支撑配方继续学习](../artifacts/runs/diagnostic_pose_learning/wbc_support_learning/README.md)原定1000轮，已根据持续退化提前结束：实际记录336轮、33,030,144 transitions，最后50轮非终止奖励99.872%裁为零，平均位置／朝向误差.25451m／1.44972rad；课程放宽后仍未恢复足够奖励信号。这支持停止本次运行，不证明跑满1000轮必败。采用最新已保存的`checkpoint_000300.pt`（301轮，后35轮未保存），没有挑选最佳检查点；SIGINT后的OS退出0不代表计划正常完成。现已进入独立64、workspace4、static16各双引擎后测，保留原策略对照与实际状态回放。候选使用UMI启发的正向足力分配和髋—足平面偏好，结合柔性限位、有效PD均值正则；原Actor、数据、PD和终止规则保持。UMI实测数据不是此轮依赖；具体技术路径依据实际证据调整，见[当前目标与推进依据](plan.md)。

单个强化学习 Actor **仅跟踪世界系末端位姿轨迹（位置＋朝向）**，统一输出 12 腿＋6 臂关节位置目标，自主决定全身运动。无外部底盘速度命令；参考 MLM 的因果目标历史与可选预测，后续面向手持 UMI。基座速度仅作估计量或 Critic／训练标签。276 维位姿观测、schema-2 轨迹／bundle、Isaac/MuJoCo 和视觉 IO 已实现并完成 CPU 检查；旧 246 维位置策略不能加载为新位姿策略。原位置验收数值不变，朝向验收阈值仍待确定。

当前工作目录为 `artifacts/runs/diagnostic_pose_learning/`。`eval_initial8/operator_summary.json` 记录 fresh 初始化策略的 **8×20 秒 PhysX 批量前测，墙钟 49.73 秒**；全部完成且无跌倒，移动引用仍有约 38.5 cm 平均位置 RMSE。这是工程前测，不是学习结果或硬件有效性证明。

fresh 276-input、single-18 Actor 的 **1000 次迭代训练已完成，退出码 0**：`train_pose_config.json` 使用 4096 环境、24 步、5 epoch、4 minibatch、stage 1；现有 family 采样混入 `references/train/` 的 8 条 EE pose 引用。`train_initial_checkpoint.pt` 的腿 action std 为 .3（.2 rad scale 下等于 .06 rad），臂目标角 std 为 .01 rad；辅助预测／速度估计、自适应采样和 DR 均关闭。`train_pose1000_summary.json` 记录 98,304,000 transitions、20,000 次优化器更新，训练循环 3434.39 秒，有限性检查全部通过；累计 **1,780 次跌倒、98,519 次重置、29,739／17,694,720,000 个子步关节样本力矩饱和**，不能用后测零跌倒覆盖这些训练事件。训练入口沿用 `scripts/train.py --initialize-from`；已完成输出由唯一 GPU operator 管理，勿重复启动。

PhysX 后测有效结果为 `eval_post1000_openblas1/report.json`，与前测的比较见 `paired_pose1000.json`：前后均完整 **8×20 秒、零跌倒**，墙钟分别 49.73／50.07 秒。局部4例位置 RMSE 均值 **.02013 → .01557 m**、朝向 **.08539 → .15053 rad**；移动4例位置 **.38503 → .06573 m**、朝向 **.09385 → .28847 rad**。全部8例位置改善、全部8例朝向退步；总体位置 **.20258 → .04065 m**、朝向 **.08962 → .21950 rad**。局部／移动平均基座平面位移由 **.00469／.00452 m** 增至 **.05426／.11932 m**；位移增加本身不是成功标准。结果支持位置跟踪改善，不能宣布完整位姿跟踪成功。朝向奖励权重对照及两组双引擎后测现已完成，结果见下。

独立复核同批已保存的50 Hz观测/FK轨迹，支持移动案例伴有下沉、倾斜和足端移动；此前承重标签依赖历史body顺序假设，该批未记录实际runtime顺序，不能据此确认承重拖移或交替迈步。球形足滚动贡献未分离，MuJoCo旧trace未保存接触，详见[验证记录中的足端重建证据](validation.md#pose1000-foot-motion)。

本轮 `orientation_tracking=1` 控制组与 `4` 候选组均已完成：同一 `checkpoint_000999.pt` 经 `--initialize-from`、fresh Adam、初始学习率 `1e-5`、seed 0，各 **250轮＝24,576,000 transitions／5,000次优化器更新**。实际配置仅朝向权重不同，run 中 source 哈希、输入、资产和临时参数一致；Git commit/dirty 描述不同。两组所有有限性检查通过，训练循环 **842.79／840.75秒**，训练跌倒 **217／240次**、重置均 **24,581次**，力矩饱和 **88,745／276,114** 个样本（每组分母 **4,423,680,000**）；评估零跌倒不覆盖这些事件。唯一 operator 已顺序完成训练、PhysX 与 MuJoCo test8，README 记录各运行退出码0；本次两组训练／评估已结束。

对照输出见 `artifacts/runs/diagnostic_pose_learning/orientation_weight_comparison/README.md` 和逐例报告。PhysX 控制→候选平均位置 RMSE **.035426 → .056234 m**，朝向 **.168073 → .119565 rad**，8例朝向全改善、位置全退步。MuJoCo 为位置 **.062277 → .060091 m**（4好4差）、朝向 **.227390 → .137394 rad**（8好）；四批评估均完整8×20秒、零跌倒。控制组本身也优于1000轮 PhysX后测，不能把继续学习的收益全部归给新权重。结果支持朝向改善可迁移，但尚未同时保住PhysX位置精度，不能宣布完整位姿成功；朝向验收阈值未新增，所有bundle仍为 `trained=false`。

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

真实组合几何的固定目标图像闭环已完成：权重1控制组bundle通过腕部渲染RGB-D→ArUco／深度测量→延迟队列→单18关节策略，在MuJoCo运行 **20秒**，退出码0、墙钟 **27.38秒**。601帧全部检出，600个独立测量送入控制，**999/1000**控制步使用有效测量；仅首步等待初始图像，此后无失跟／保持、未触发跌倒。2秒后真实TCP位置／朝向RMSE为 **.018855 m／.085240 rad**，图像目标测量RMSE为 **.038886 m／.066913 rad**。世界相机定位使用仿真里程计，控制目标仅来自图像；这是固定目标、无注入遮挡的工程闭环，未验证移动目标、真实相机或60秒视觉任务。详细场景、输入哈希、20秒腕视角视频和 `run.py probe/run` 复现命令见 [图像闭环README](../artifacts/runs/diagnostic_pose_visual_control/README.md)，复现按该入口使用新输出目录。

此前建议的CPU配对回放已完成（退出码0）：复用 `orientation_weight_comparison/` 两组bundle及 `eval_control/`、`eval_candidate/` 原始trace，`clip(raw)` 与保存action最大差 **1.55e-6**。在两组moving4的 `t >= 2 s` 保存状态上，腿raw动作越界比例 **90.28% → 90.87%**，臂 **4.79% → 3.59%**，臂均仅J5越界；基座倾斜变化混合，未见权重4新增普遍裁剪。两组都存在腿动作顶边，不能据此把位置代价单独归因于候选新增裁剪；足路径仅报未接触门控的FK运动，不新增承重／步态结论。这是保存状态关联证据，不是奖励或探索的动态因果证明。

**联合位姿奖励候选已完成训练及双引擎评估，均退出0**：`2*rpos+rrot → 3*rpos*rrot` 保持 `.15 m／.5 rad` 宽度和峰值3，但不保持梯度；源代码 `d26b514` 已review、12项CPU测试通过。复用w1控制组，从同一原 `checkpoint_000999.pt` 经fresh Adam、初始LR `1e-5`、seed0完成 **250轮／24,576,000 transitions／5,000次更新**，其余配置固定。训练循环 **845.19秒**，全部有限；**297次跌倒、24,589次重置、110,112／4,423,680,000** 个子步关节样本饱和，不能由后测零跌倒覆盖。

PhysX控制→联合奖励位置RMSE **.035426 → .038529 m**（2好6差）、朝向 **.168073 → .128677 rad**（8好）；MuJoCo位置 **.062277 → .037347 m**、朝向 **.227390 → .159585 rad**（各7好1差）。候选两引擎各8×20秒完整、零跌倒，现有仅位置pass均7/8，位姿验收仍为null。相较权重4，联合奖励的PhysX位置代价较小，但朝向改善也较少，尚未实现各例两类误差同时改善。训练、逐例／分组比较与复现命令见 [coupled_pose_comparison/README.md](../artifacts/runs/diagnostic_pose_learning/coupled_pose_comparison/README.md)。

新候选8份PhysX trace实际保存30个Unicode `contact_body_names`，与30列足／身体净接触力对齐，`allow_pickle=False`可读，见 `coupled_pose_comparison/contact_names_check.json`。列名直接来自本次runtime；这验证了实际保存路径，不能追认旧trace的body顺序，也不证明接触对动力学或步态。

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

旧 `artifacts/evaluation/suite` 的100例均缺朝向，当前pose `load_suite`拒绝，尚未实跑。当前CLI生成schema-2位姿套件；新增朝向采样改变随机数消耗，相同seed不证明位置数组等同旧套件。新生成必须另选未使用目录，保留旧文件；下例 `suite_pose_new` 仅作新目录示例。

```bash
conda activate pawweaver-data
python -m pawweaver.evaluation create artifacts/evaluation/suite_pose_new
conda activate pawweaver-train
python scripts/evaluate_isaac.py --asset assets/generated/verified --bundle artifacts/runs/baseline_seed0/bundle --suite artifacts/evaluation/suite_pose_new --seed 0 --output artifacts/evaluation/physx_seed0 --headless
conda activate pawweaver-runtime
python scripts/evaluate_mujoco.py --asset assets/generated/verified --bundle artifacts/runs/baseline_seed0/bundle --suite artifacts/evaluation/suite_pose_new --seed 0 --output artifacts/evaluation/mujoco_seed0
python -m pawweaver.evaluation compare artifacts/evaluation/physx_seed0/report.json artifacts/evaluation/mujoco_seed0/report.json --output artifacts/evaluation/comparison_seed0.json
```

默认 100 回合，含 0.3–2 m 静态目标与 60 秒动态轨迹。候选集固定随机种子和文件哈希，尚未完成精确硬件可达性审核，因此不能直接给出验收结论。每个训练种子都须使用同一策略在两个引擎评估。

## FastUMI 与视觉

在 `pawweaver-data` 中运行 `python -m pawweaver.data --help`。输入需要 pose、逐帧时间戳、稳定 source ID，以及单位、sensor→TCP、source→task、速度/加速度、工作区边界的配置。转换器组合完整 source→task、输入 pose、sensor→TCP 刚体变换；位置和 WXYZ 朝向使用相同时间缩放，朝向以 SLERP 重采样。转换器不下载视频，增强前按 source ID 划分数据。原FastUMI本地样本已取得并检查，正式世界TCP示范转换尚未完成，当前训练仍使用上文合成FK引用。

用户已手动下载原FastUMI `close_ricecooker.tar.gz`：**3,437,750,379字节**与固定官方列表大小一致，完整gzip CRC／tar目录检查通过，本地SHA256已记录；官方SHA在既有元数据中被遮蔽，未声称官方哈希匹配。包内为20个HDF5、无CSV或config／标定侧文件。按归档顺序选取并仅解出首个 `episode_17.hdf5`，实查qpos/action均为 **120×7**、有限且完全相同，图像为 **120×1080×1920×3**。见 [本地样本检查](../artifacts/data/fastumi_original_sample/LOCAL_SAMPLE.md) 及同目录 `local_archive_integrity.json`、`episode_inspection.json`、`local_sample_result.json`。

该episode缺少原始逐帧时间、frame／原点、单位及适用标定，不能从120行推定原时长，也不能仅凭七列和四元数范数确认sensor或已处理TCP身份；官方producer约定支持XYZW，但未绑定本文件处理状态。正式转换仍需该episode对应的时间与语义／刚体变换，不套用通用Xarm6示例。此前读取的 [FastUMI Pro文本样例](https://huggingface.co/datasets/LumosRobotics-FastUMIPro/example_data_fastumi_pro_raw/resolve/c3e3d1c4ca25ea32cc19e50635d0d13af2ccef6b/task2/session_001/Merged_Trajectory/merged_trajectory.txt) 是另一来源，不能替代原FastUMI的标定或时间；两者均未作为正式TCP示范导入训练。

原episode17另已派生20秒schema-2工程命令，CPU构造／加载检查及coupled250 MuJoCo评估均退出0；完整20秒、无跌倒，2秒后位置／朝向RMSE **.049607 m／.155485 rad**。使用人为 `20*i/119` 重定时、米制XYZW工程假设和一次固定 `G*T0^-1*Ti` 对齐；未知sensor→TCP外参仍未消除，不是原始时序或正式TCP示范转换，未加入当前训练或冻结test8。见 [派生命令](../artifacts/data/fastumi_original_sample/derived_command20/README.md) 与 [实际MuJoCo结果](../artifacts/data/fastumi_original_sample/derived_command20/mujoco_coupled250/README.md)；此新例尚无同例PhysX或视觉闭环证据。

早期 **HTTP 401／GatedRepo**、正常配置token文件缺失，以及授权后Chrome CDN跳转的 `ERR_BLOCKED_BY_CLIENT` 保留在 [访问记录](../artifacts/data/fastumi_original_sample/README.md) 和对应JSON中；它们是历史获取失败，已由上述本地下载事实更新，不再代表当前没有文件。

在 `pawweaver-runtime` 中运行 `python -m pawweaver.visual_runtime --help`。需要有效资产、策略包、轨迹与 scenario JSON。scenario 定义 `marker_id`、`marker_size_m`、`marker_to_goal`（标记系平移）、`marker_to_goal_quat_wxyz`（显式标记到目标朝向标定），可加入延迟、遮挡、深度缺失、位置噪声和随机种子；`--record` 输出腕部视频。

相机在 2 ms 物理时钟上调度约 30 Hz 图像，50 Hz 控制器消费到达的测量。真值仅用于场景生成和评分。持续失跟后暂停参考推进、保持有界关节目标；这种保持的整机稳定性仍需训练策略验证。基座定位来自仿真状态，不包含真机自主定位。
