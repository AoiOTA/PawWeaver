# PawWeaver · 四足机械臂全身协同

PawWeaver以AS2 EDU＋Piper-H为对象，用一个强化学习Actor统一输出12腿＋6臂关节位置目标，通过显式PD控制全身。目标是根据世界系末端位姿轨迹自主协调腿、臂与机身，必要时下蹲、倾身、调整支撑和移动。Isaac Lab／PhysX训练，独立MuJoCo验证，状态控制稳定后接入图像目标。

当前使用RealSense D435与Piper-H腕部支架；原DC1配置保留。全部工程学习仍使用明确标注的临时参数，尚无通过完整全身任务验收的策略。

## 当前进展

2026-09-10的v3改进方案正在执行。完整评估时长、正常停止时保存最后完整更新、实际奖励分项和固定组采样已接通。取消非负裁零的受控实验改善了位置跟踪；连续1000轮世界系课程仍存在持续折足、支撑转换失败与精度退化。单项折足惩罚没有一致收益，未采用。

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

PawWeaver与`/home/lyb/kiss-my-agent-dogfood`是本轮并行推进的两个目标。KMA针对实际工作中的分工、代理复用和机械执行阻塞更新指导，并保留原v0.2.7角色的移除兼容性；候选`0.2.7+codex.20260910132339`已部署，13个候选文件与安装缓存一致，插件／仓库验证、18项setup检查及独立审查通过。后续实际工作正在使用这些指导；当前证据证明安装与静态兼容，尚不能证明效率提升或新会话全部角色已刷新。

## 使用与边界

使用既有`pawweaver-train`、`pawweaver-runtime`和`pawweaver-data` Conda环境，不向系统或base环境安装项目软件。环境说明见[environments](docs/environments.md)，实际训练、评估与复现命令见[runbook](docs/runbook.md)，当前目标与实验依据见[plan](docs/plan.md)。

- 固定40mm夹爪开度；初版不含抓取接触、真机控制或自主全局定位。
- 控制周期20ms、物理步长2ms；关节顺序FR／FL／RR／RL各hip/thigh/calf，再接arm_joint1…6。
- 四元数统一WXYZ，采用SI单位；相机实际标定与虚拟参数分开记录。
- 同一时刻由一个operator拥有GPU仿真及其输出目录。

组合机器人已有URDF、USD和MJCF。资产审核可在`pawweaver-runtime`运行`pawweaver audit`，M0未闭合时退出2；未覆盖原始模型。大型轨迹、权重和视频留在既有工作区，仓库保留必要配置、结果与定位路径。

远程origin已配置为AoiOTA/PawWeaver，先前授权的发布记录保留。本次v3改动仅作本地独立提交，未推送；KMA改动保留在自己的仓库。
