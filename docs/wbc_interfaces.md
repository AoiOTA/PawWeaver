# WBC 输入接口与 VLA 对接

这里的“WBC 输入”须分成两层：**高层提供的任务指令**与**系统提供的本体状态**。策略将两者组合成观测，输出关节动作；Actor 的观测维度不是 VLA 的动作维度。6DoF 位姿也不等于6个数：XYZ＋四元数为7维，XYZ＋rotation6D为9维。以下只比较接口，不移植论文验收标准。

## 代表性接口

| 工作 | 高层交给控制器的任务指令 | 坐标系、编码与时间 | 底层动作 |
|---|---|---|---|
| **UMI on Legs** | 仅TCP位姿轨迹；无外部底盘速度命令 | 固定task/world系；每帧XYZ＋rotation6D，8帧＝72维目标信息，包含历史、当前与未来目标 | 一个策略输出12腿＋6臂关节目标，50 Hz，经PD执行 |
| **MLM** | 仅世界系EE 6DoF轨迹；底盘速度是估计量／训练信息 | 每帧9维；`t−3:t`四帧含当前，另有` t+1:t+4`四帧未来目标；TVP/NAE预测未来，也可消费DP直接给出的未来轨迹 | 一个18维关节偏移策略，50 Hz，经PD执行 |
| **RoboDuet** | EE位姿＋`vx, vy, yaw_rate` | EE参考随base XY/yaw移动，高度取地形高度＋固定参考高.38 m，不随实际base高度、roll/pitch变化；位置为球坐标；论文pose9维，代码有6／9维分支 | 两个合作策略：臂策略输出6臂关节偏移＋2维身体姿态引导；腿策略输出12腿关节偏移，50 Hz |
| **Deep Whole-Body Control** | EE位姿＋`vx, yaw_rate` | EE位置为球坐标，参考随base XY/yaw移动、参考z固定.53 m；不是固定世界目标。遥操作先累加目标增量，再送给控制器 | 一个策略统一输出12腿＋6臂关节目标，50 Hz，经PD执行；夹爪另控 |

UMI on Legs 的8个目标采样相对当前时刻为 **−60、−40、−20、0、20、40、60、1000 ms**，不是均匀的8帧未来动作；上层预测在进入WBC前转换到固定任务坐标系。见[论文§3.2、Fig.3](https://umi-on-legs.github.io/static/umi-on-legs.pdf)、[官方仓库](https://github.com/real-stanford/umi-on-legs)。MLM 将“没有真实未来数据的遥操作”与“DP直接提供未来轨迹”分开处理；NAE的预测不能当作已观测的未来。见[论文§III-A/B、Fig.2、Eq.2](https://arxiv.org/html/2508.10538v2)。

RoboDuet 的论文写位置球坐标3维＋旋转6维；官方代码默认为位置3维＋轴夹角3维，`use_rot6d` 分支才是9维。不能把“6DoF”“6个浮点数”和rotation6D混为一谈；也不能把这种随底盘移动的目标当作固定世界轨迹。见[论文§III](https://arxiv.org/html/2403.17367)、[目标变换及编码代码](https://github.com/locomanip-duet/RoboDuet/blob/master/go1_gym/envs/automatic/legged_robot.py#L810)。

DeepWBC 的球坐标增量是遥操作目标生成方式，不是18维RL动作的含义；夹爪开闭由手柄或脚本独立控制。论文与公开代码的状态维数、base命令槽数有差异，因此不混列总维数：论文明确的外部速度命令是前进速度与偏航角速度。见[论文§2、§3.3及附录C/D/F](https://arxiv.org/html/2210.10044v1)、[官方观测代码](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/main/legged_gym/legged_gym/envs/widowGo1/widowGo1.py#L871)。

## 本体状态不是高层动作

上述WBC还消费关节位置／速度、基座姿态或重力方向、角速度、上一步动作等信息；这些来自机器人感知与估计。UMI on Legs 使用机器人本体状态；MLM使用5帧本体历史并估计基座线速度；RoboDuet两策略有各自的观测／历史和协作信息；DeepWBC还使用足接触与适应模块的环境latent。**VLA输出任务目标，不负责伪造这些状态。** 本体状态中出现base velocity，也不意味着高层必须输出base velocity command。

## 两个容易混淆的名字

**FastUMI 是采集与模仿学习体系，不是四足WBC。** 原始位姿为`[x,y,z,qx,qy,qz,qw]`，另带时间信息；经标定／对齐可以构造TCP轨迹。DP相对TCP动作、ACT关节动作及其他分支并非统一接口，夹爪数据也不能直接塞入18维腿臂策略。为世界位姿WBC选用位姿分支时，仍需处理初始参考系、camera→TCP、时间及四元数顺序；机器人专用IK关节数据不是通用EE指令。见[论文与数据处理说明](https://arxiv.org/html/2409.19499)、[正式论文入口](https://proceedings.mlr.press/v305/zhaxizhuoma25a.html)。

**QUAR-VLA 不带机械臂。** WR-2是12腿关节四足，所谓操作包含背球倾倒。高层以约2 Hz输出11维命令＋终止标志：`vx, vy, ωz`、3个步态参数、步频、身体高度、pitch、足间宽度、抬脚高度、terminate；下游为Walk These Ways命令跟踪策略，而非EE位姿WBC。原实验低层准确观测维度、频率及速度坐标系未在本次核实，不从后续QUART-Online代码补填。见[论文§3.1–3.3及参考文献23](https://arxiv.org/html/2312.14457v6)、[官方项目页](https://quart-robot.github.io/)。

## 对 PawWeaver 当前接口的含义

PawWeaver遵循**仅固定世界系TCP位置＋朝向轨迹**这一任务边界，没有外部base速度命令。VLA侧适合提供带时间的世界TCP位姿目标；若输出相机系或相对动作，需先经适配层做坐标变换、增量还原与时间采样。这是对接关系说明，不代表现有仓库已经实现完整VLA适配器。

当前轨迹入口为时间戳、XYZ米制位置和**WXYZ**四元数；位置线性插值、朝向最短路径SLERP。在线目标入口还消费有效性、置信度及时间戳。系统将本体状态、上一动作、当前TCP信息和目标历史组装为 **276维Actor观测**，内部姿态采用rotation6D；**276维不是VLA输出**。一个Actor以50 Hz生成12腿＋6臂关节位置目标，夹爪不在18维中。见[轨迹实现](../src/pawweaver/trajectories.py)、[观测构造](../src/pawweaver/observations.py)。

当前运行配置是**因果4帧目标历史、轨迹预测关闭**。即使VLA生成未来chunk，现有路径也不等于将未来4帧直接送入Actor；不能宣称已具备MLM的完整future-preview能力。世界目标在观测构造时转到当前base系只是数值表达，不会让目标随base重新锚定。本文不改变该接口、增加schema或扩展科研验收。
