# 固定末期奖励目标核查（最后一步，已结束）

在本次相同 alpha25 配置、相同三个任务与槽位、相同末期奖励宽度下，checkpoint325 的三个任务实际非终止奖励均低于 neutral checkpoint499。**本次没有支持“退化行为反而获得更高配置奖励”；固定评价目标本身也退步。** 这只是确定性策略、固定末宽度的条件比较，不能证明 PPO 更新方向、唯一训练根因或完整混合随机训练回报。全身能力和硬件验收均未完成；不继续训练或新增测试。

## 配对结果

两策略各 3 个案例均完整执行 3000 步／60 秒，无 fall，终止奖励均为 0。奖励采样目标时间为 k×0.02；command 使用 [8,60) 的2600步，hold 使用 [20,60) 的2000步。表中奖励为实际非终止项的平均 rate（乘0.02才是单步奖励），位置和速度采用同一窗口。完整 reward 分项、dt缩放总数及前后差值见 `reward_readout.json`。

| 任务 | 8–60s reward rate 旧→新 | 20–60s reward rate 旧→新 | 20–60s EE位置RMSE m 旧→新 | 20–60s实际 vx m/s 旧→新 |
|---|---:|---:|---:|---:|
| neutral_forward | 6.36931 → 5.65393 | 6.36999 → 5.65574 | 0.00824 → 0.01122 | 0.198740 → 0.201162 |
| neutral_backward | 6.55062 → 0.58303 | 6.55069 → 0.54430 | 0.00819 → 0.05621 | -0.150847 → -0.000074 |
| high_stand | 2.58507 → 2.31915 | 2.34501 → 2.10926 | 0.14304 → 0.21067 | -0.000173 → 0.001104 |

| 任务 | 旧策略0–60s实际总reward | 新策略0–60s实际总reward | 旧／新终止项 |
|---|---:|---:|---:|
| neutral_forward | 383.88415 | 344.02031 | 0 / 0 |
| neutral_backward | 392.76554 | 78.59564 | 0 / 0 |
| high_stand | 188.07427 | 171.88678 | 0 / 0 |

- 后退：20–60s奖励rate下降6.00639。主要分项变化为 collision −2.00000、UMI pose −1.94495、线速度追踪 −1.25765、feet_under_hips −0.84582。实际后退消失；新策略保持期所有采样均有 RL_thigh 净接触力 >5 N，四足同时受力。这是非足接触的失败行为，不能因未触发fall称为成功支撑。
- high：位置RMSE从0.14304增至0.21067 m；姿态RMSE则从0.14932降至0.09254 rad。总奖励rate仍下降0.23575，主要为 feet_under_hips −0.20739 和 UMI pose −0.04909。
- 前进：本次新策略仍有约0.20116 m/s 的yaw-frame前进速度，**没有复现“前进静止”**；其yaw rate RMSE从0.03522增至0.20030 rad/s，feet_under_hips及yaw追踪项分别下降0.30072和0.29539。它与此前11环境评估的环境数、槽位布局不同，不替换此前证据，也不做跨批次因果归因。

两策略前进/high 的保存采样均未见机器人非足净力 >5 N；旧后退也未见，新后退如上。既有 full-FK/support 消费者直接读取原轨迹，最大TCP世界位置重建误差旧2.789e−6 m、新1.866e−6 m。support既有post20窗口按保存后状态时间包含20.00s边界（2001样本），奖励与任务表严格按目标时间2000样本；不混作相同采样边界。没有fall片段需要单列。

## 采集及复现

`evaluate_reward.py` 为既有 `scripts/evaluate_commanded_isaac.py` 的本目录局部副本，保存原动作、目标注入、obs、轨迹、终止、suite/policy身份和support输入。只把评分配置统一为alpha25，并打开 `reward_diagnostics=True`。每次reset后载入同一个STOP325 checkpoint的 `umi_pose_reward`，实际位置sigma为 **0.005 m²**、姿态sigma为 **0.5 rad**；EMA正常更新，在`auto_reset=False`下不改变宽度。逐步断言宽度固定，立即clone实际 `env.step` 第二返回reward和所有 `weighted_nonterminal_terms`、preclip/postclip。逐步校验分项和≈preclip、关闭clipping时preclip=postclip，以及reward≈0.02×postclip−5×fallen；保存数据重建reward最大差为0。第一fall帧会含入，其后该slot不再记录。案例身份仅使用manifest固定slot，不使用demonstration group统计。

策略/源checkpoint/原评估器哈希在 `provenance.json`，状态在 `score_state.json`，三案例在 `suite/manifest.json`。EE任务坐标系是 `base_xy_yaw_ground_z`，速度命令来自预设轨迹；并非world-fixed EE-only验收。未改变通用源码、原checkpoint或bundle。

```bash
bash artifacts/runs/diagnostic_pose_learning/plan_v3_reward_objective_probe/run.sh old 2
bash artifacts/runs/diagnostic_pose_learning/plan_v3_reward_objective_probe/run.sh new 1
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /home/lyb/miniconda3/envs/pawweaver-runtime/bin/python artifacts/runs/diagnostic_pose_learning/plan_v3_reward_objective_probe/readout.py
```

保留现有日志时run.sh拒绝覆盖，不应重跑。两次实际成功命令与退出码各有独立文件。原始trace和console日志保留本地；提交配置/入口/身份/结构化读出和report即可。

## 执行失败与停止边界

首次old在Kit startup、尚未创建环境或采样时SIGSEGV，退出139；原生栈为getenv → libxcb `_xcb_parse_display` → XOpenDisplay → omni.platforminfo/telemetry。这只定位崩溃路径，不确定根因。完整日志保留 `old_attempt1_console.log`。随后经root授权一次完全相同配置/命令的启动重试，`old_attempt2`成功退出0，再运行`new_attempt1`成功退出0；未修改环境变量/Kit设置。首次失败不会因重试成功被删除或称作首次稳定。

CPU readout首个尝试因末行print缺右花括号而SyntaxError退出1；在本目录局部修正后，读取相同保存轨迹成功退出0，未重跑仿真。失败与成功日志/退出码分别保留。没有额外测试套件、模拟或训练。两个GPU进程均真实退出，`nvidia-smi`确认无compute进程；GPU已释放。完成本步后按用户要求两仓提交推送并暂停，由root统一处理发布。
