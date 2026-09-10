# PawWeaver · 四足机械臂全身协同

PawWeaver以AS2 EDU＋Piper-H为对象，用一个强化学习Actor统一输出12腿＋6臂关节位置目标，通过显式PD控制全身。目标是根据世界系末端位姿轨迹自主协调腿、臂与机身，必要时下蹲、倾身、调整支撑和移动。Isaac Lab／PhysX训练，独立MuJoCo验证，状态控制稳定后接入图像目标。

当前使用RealSense D435与Piper-H腕部支架；原DC1配置保留。全部工程学习仍使用明确标注的临时参数，尚无通过完整全身任务验收的策略。

## 当前进展

**当前推进（E3之后）**：已明确安排新的[B路线中立末端学习前置实验](artifacts/runs/diagnostic_pose_learning/plan_v3_neutral_learning/README.md)，不是旧E3原样自动续训。fresh279输入、单18关节Actor，只使用原有7条neutral轨迹，站立／移动采样30/70；初始腿／臂std为.25/.02rad、std可学习、entropy .01。独立连续500轮预算为4096×24、49,152,000次交互／10,000次更新；训练正在运行，250检查点的MuJoCo实际开发读出尚待完成（neutral7为训练来源，非独立测试集），训练均值不证明命令移动。`b_neutral_learning`是本次唯一GPU operator；先判读实际速度、末端误差和接触支撑，若形成有支撑的命令移动再返回完整末端任务组合，不自动加预算。

[固定关节参考PD保持探针](artifacts/runs/diagnostic_pose_learning/plan_v3_pd_hold_probe/README.md)已实际退出0：两组均运行20秒且未触发既有跌倒判据，但都未保持目标。kp30修复了局部joint3力矩缺口，TCP误差却从1.153增至1.403m，首次非足接地由8.28提前到5.28秒；不采用为任务修复。该局部结果不改变当前训练PD。

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

PawWeaver与`/home/lyb/kiss-my-agent-dogfood`持续作为同等重要的两个目标推进。KMA已完成`8931a47`的分工／复用与兼容修正，后续`a217b97`补充组件探针应保留下游约束语义；最新部署为`0.2.7+codex.20260910135927`。后者源于worker提出绕过约束的方案，由root在测试前纠正，不能宣称指导已自主预防此问题。文档与视频等真实交付已使用更新后的分工指导；这证明本轮执行，不证明量化效率提升、新会话角色全部刷新或WBC成功。

## 使用与边界

使用既有`pawweaver-train`、`pawweaver-runtime`和`pawweaver-data` Conda环境，不向系统或base环境安装项目软件。环境说明见[environments](docs/environments.md)，实际训练、评估与复现命令见[runbook](docs/runbook.md)，当前目标与实验依据见[plan](docs/plan.md)。

- 固定40mm夹爪开度；初版不含抓取接触、真机控制或自主全局定位。
- 控制周期20ms、物理步长2ms；关节顺序FR／FL／RR／RL各hip/thigh/calf，再接arm_joint1…6。
- 四元数统一WXYZ，采用SI单位；相机实际标定与虚拟参数分开记录。
- 同一时刻由一个operator拥有GPU仿真及其输出目录。

组合机器人已有URDF、USD和MJCF。资产审核可在`pawweaver-runtime`运行`pawweaver audit`，M0未闭合时退出2；未覆盖原始模型。大型轨迹、权重和视频留在既有工作区，仓库保留必要配置、结果与定位路径。

远程origin已配置为AoiOTA/PawWeaver，先前授权的发布记录保留。本次v3改动仅作本地独立提交，未推送；KMA改动保留在自己的仓库。
