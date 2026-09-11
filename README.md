# PawWeaver · 四足机械臂全身协同

PawWeaver以AS2 EDU＋Piper-H为对象，用一个强化学习Actor统一输出12腿＋6臂关节位置目标，通过显式PD控制全身。目标是根据世界系末端位姿轨迹自主协调腿、臂与机身，必要时下蹲、倾身、调整支撑和移动。Isaac Lab／PhysX训练，独立MuJoCo验证，状态控制稳定后接入图像目标。

当前使用RealSense D435与Piper-H腕部支架；原DC1配置保留。全部工程学习仍使用明确标注的临时参数，尚无通过完整全身任务验收的策略。

## 当前进展

[neutral500](artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/README.md)已完成：两引擎均7/7完整60秒，出现有足支撑的前后、左移、转向与弧线运动，末端保持位置误差约.4–.9cm；PhysX右移仍静止，MuJoCo右移有效。可查看[完整60秒后退跟随视频](artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/video_replay/final500_neutral_backward_follow.mp4)，这是保存状态回放。

[原28末端任务迁移](artifacts/runs/diagnostic_pose_learning/plan_v3_full_pose_transfer/README.md)的500轮及全部双引擎后测已结束，候选不采用：低位保持位置误差虽改善至约32cm，高位仍约90cm，侧向仍跌倒；两引擎全部neutral案例的末端位置／朝向精度退化，PhysX前进和后退近静止。保留原neutral500，不原样扩训或启动独立test8。

[25%末端幅度课程](artifacts/runs/diagnostic_pose_learning/plan_v3_pose_amplitude25/README.md)在250检查点暴露技能遗忘后，于326轮正常保存停止：32,047,104次交互、最终checkpoint325。双引擎终测各11/11完整60秒，但后退、侧移与转向严重退化；后退保持窗持续非足接触，前进伴随约+.18rad/s非指令转向。低位误差部分改善，高位与侧向位置误差增大，候选不采用。

[归一化统计量交换诊断](artifacts/runs/diagnostic_pose_learning/plan_v3_normalizer_closed_loop/README.md)的四个CPU回合已完成：新权重换回旧统计量没有恢复移动，后退27.88秒跌倒；旧权重配新统计量仍保留大部分前后移动。结果不支持统计量单独造成遗忘，也没有确立唯一训练根因。最后的[固定奖励对照](artifacts/runs/diagnostic_pose_learning/plan_v3_reward_objective_probe/README.md)记录实际奖励分项与任务表现。

最后六例固定奖励对照已完成，旧／新各3个60秒任务，全部实际退出0。20–60秒加权非终止奖励均值（乘dt前）为前进6.36999→5.65574、后退6.55069→.54430、高位2.34501→2.10926；本批新策略前进仍移动，后退近零，高位位置误差增大。三项奖励均降低，不支持在这些固定任务上以更高配置奖励解释行为退化，也不能据此确定PPO更新或训练退化的唯一原因。首次Isaac原生启动崩溃exit139，原样单次重试后完成；本地读出语法错误已修复并读取原轨迹成功。失败记录保留，GPU已释放。

**收尾边界（2026-09-11）**：按用户要求完成当前诊断、提交推送两个仓库后暂停；不自动续训、追加实验或修改KMA规则。原完整全身控制与KMA实际收益目标继续保留为未完成。以下v3记录为本阶段已执行的历史，后续推进需用户重新提出。

完整评估时长、正常停止保存最后完整更新、实际奖励分项和固定组采样已接通。取消非负裁零的受控实验改善位置跟踪；连续1000轮世界系课程仍有持续折足、支撑转换失败与精度退化。单项折足惩罚没有一致收益，未采用。

速度命令＋末端目标B路线已完成fresh1000轮，98,304,000次交互／20,000次更新全部有限，3987次训练跌倒。初始／最终双引擎评估均已退出0：初始各组8/8完整60秒；最终PhysX开发／独立测试为7/8、4/8，MuJoCo为7/8、1/8。完整命令窗中实际移动速度接近0，非中立末端任务仍有大误差，部分存活回合也有非足接触。该预算内没有形成可用移动操作策略，不采用为成果；不据此自动加训或进入条件性E4／正式E5。

| 路线 | 任务输入与坐标 | 能够回答的问题 |
|---|---|---|
| A：世界系EE-only，276输入 | 固定世界系末端位置＋朝向；没有外部底盘速度命令 | 能否自主为末端任务协调全身与支撑移动 |
| B：速度＋EE比较，279输入 | 预设yaw系vx/vy/yaw变化率；EE原点随base XY、旋转随base yaw，地面Z固定，不随base高度／roll／pitch | 能否按给定移动命令完成全身操作协同 |

B路线仿真速度来自预设任务，未来遥操作由操作员提供，真实部署尚未实现。它是整套方法配置比较，不是额外3维输入的单因素消融，也不能证明世界固定EE-only成功。两条路线都保留单Actor；旧246维位置策略不实现完整位姿接口。

本次主要证据：

- [奖励信号修复与受控结果](artifacts/runs/diagnostic_pose_learning/plan_v3_reward_signal/README.md)
- [世界系课程连续学习](artifacts/runs/diagnostic_pose_learning/plan_v3_task_curriculum/README.md)
- [折足成本的负结果](artifacts/runs/diagnostic_pose_learning/plan_v3_foot_fold/README.md)
- [速度＋末端目标比较](artifacts/runs/diagnostic_pose_learning/plan_v3_commanded_pose/README.md)

历史实验、失败与原始路径保留在[运行手册](docs/runbook.md)、[验证记录](docs/validation.md)及各实验目录；本地Git历史完整保留。

## 仍需完成

低位、高位、侧方、远距与变化朝向任务下的主动下蹲、倾身、合理支撑转换、迈步及腿臂双向补偿仍未达成。临时几何或短轨迹不能建立硬件有效性，所有工程bundle仍为`trained=false`；原性能指标保留，正式朝向阈值待用户确定。

已有20秒固定图像目标闭环是有限条件的工程证据，尚未完成移动视觉目标、失跟恢复或60秒视觉任务。真实装配参数、设备映射与执行器能力仍待核实，见[硬件参数与实机准备](docs/hardware.md)。

PawWeaver与`/home/lyb/kiss-my-agent-dogfood`仍是同等重要的两个未完成目标，本阶段按用户要求一并收尾暂停。KMA最新修正为`a217b97`，部署版本`0.2.7+codex.20260910135927`。后续真实工作已复用保存轨迹完成局部修复、主动更新过时解释并缩小无必要的评估；尚未发现需要再改规则的新问题。这支持指导已用于本轮工作，不证明量化效率提升或自动预防全部问题；此前绕过约束的提议由root在测试前纠正。

## 使用与边界

使用既有`pawweaver-train`、`pawweaver-runtime`和`pawweaver-data` Conda环境，不向系统或base环境安装项目软件。环境说明见[environments](docs/environments.md)，实际训练、评估与复现命令见[runbook](docs/runbook.md)，当前目标与实验依据见[plan](docs/plan.md)。

- 固定40mm夹爪开度；初版不含抓取接触、真机控制或自主全局定位。
- 控制周期20ms、物理步长2ms；关节顺序FR／FL／RR／RL各hip/thigh/calf，再接arm_joint1…6。
- 四元数统一WXYZ，采用SI单位；相机实际标定与虚拟参数分开记录。
- 同一时刻由一个operator拥有GPU仿真及其输出目录。

组合机器人已有URDF、USD和MJCF。资产审核可在`pawweaver-runtime`运行`pawweaver audit`，M0未闭合时退出2；未覆盖原始模型。大型轨迹、权重和视频留在既有工作区，仓库保留必要配置、结果与定位路径。

本次用户已授权提交并推送：PawWeaver发布到origin/main，KMA发布自己的开发分支并按其仓库流程提交PR。保留完整本地历史；实验配置、脚本、摘要与精选结果进入Git，检查点、原始轨迹和大日志继续保存在本地实验目录。此次不创建新版本或Release。
