# E2 固定任务课程与连续学习

2026-09-10 准备完成 10 条 schema-2、60 秒、50 Hz **世界系 TCP 位置＋朝向**轨迹；准备阶段未运行训练、GPU、MuJoCo 或 `mj_step`；之后执行记录见下方，项目默认配方未改。`training_group` 为 `near/body/step`，供现有 `demonstration_group_weights` 消费。组名是训练候选用途，不是已学会的行为标签。

| 组 | 数量与训练来源 | 时间安排 |
|---|---|---|
| near | `references/train/train_local_00..03.npz` | 10 秒从实际 reset 朝向平滑接入，20 秒原局部轨迹，30 秒末端保持 |
| body | `wbc_random_paths/train_geometry_pool.json` 的 `train_0244/0018/0190` | 20 秒接近低／高／侧端点，40 秒保持 |
| step | 同一训练池的 `train_0086/0363/0081`，离线整体水平平移 | 20 秒接近，40 秒保持 |

所有轨迹从当前临时模型、UMI 配方默认关节位置和 `initial_base_height_m` 的精确 URDF FK 开始：TCP 约 `[.208346061, 0, .646208419] m`。仅保存 TCP pose 作为训练目标；离线根姿态／关节配置见证保存在生成说明，不是 Actor 命令。GoalBank 仅在回合 reset 时按起始 TCP 做一次固定平移，不随机器人底盘重新锚定。原位置 `.3 m/s` 上限和既有 `.3/tool_length = 1.256281407 rad/s` 工程角速度尺度保留；后者不是硬件限值或朝向验收阈值。接近使用五次位置插值、同参数最短路径 SLERP；原局部轨迹保持原数据与时钟。实际最大离散速度为 `.130395 m/s`、角速度 `.267027 rad/s`。

三个 body 端点在指定精确朝向下超出固定站姿腕部外界分别 `.020227/.014640/.017742 m`，**都小于原 `.05 m` 位置容差**。它们是低高侧训练任务，不能据此宣称在原验收容差内必须改变身体姿态；朝向正式验收阈值仍未指定。

三个 step 端点依次为 `[.716207,.916740,1.165614]`、`[.882608,.680603,1.214639]`、`[-.673490,.918259,1.206306] m`。它们仅对各自训练几何见证作整体水平平移，保留关节配置、高度、朝向及相对支撑几何。实际 URDF 原接触到 TCP 的保守树链长上界包含 `.028 m` 足球半径，约 `1.784596836 m`；每个新端点至少对一个原地面接触超出该界 `.060 m`。这在 `.05 m` 位置容差外仍留 `.01 m` **工程余量**，仅排除同时保持全部原接触位置；不区分迈步、滑动、抬足或失去支撑。任务仍耦合高位操作，未证明动态可达。

训练池保存的是既有静态端点几何筛查结果。本次以独立 URDF FK 核对其 TCP pose，未重新运行碰撞检查；平坦均匀地面的整体水平平移保留离线端点相对几何，但 TCP 插值路径没有经过全路径碰撞／动力学筛查。没有读取 `test_geometry_pool.json` 或 `wbc_workspace` 测试端点，也没有把测试数据改标为训练。所有参数和结果仍是临时硬件下的工程准备，不能建立硬件有效性或正式里程碑完成。

`generation_summary.json` 保存训练来源哈希、完整端点见证、平移量、链界、速度和保持时窗；`train/manifest.json` 列出这 10 个文件。真实 CPU GoalBank 消费检查使用示例概率 `{near:.4, body:.3, step:.3}`（只是本次检查，尚未写入训练配置），120 个 reset 实际抽到 `46/38/36`，覆盖全部 10 个文件；检查整个世界目标张量、四元数和非零环境原点的一次固定平移，结果见 `consumer_check.json`。

执行命令（退出 0）：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/lyb/miniconda3/envs/pawweaver-train/bin/python \
  artifacts/runs/diagnostic_pose_learning/plan_v3_task_curriculum/prepare.py
```

输出已存在时脚本拒绝覆盖 manifest。`case_id` 标识本批任务，`source_id` 则保留原始训练来源：near 沿用原轨迹 `source_id`（缺失时用原训练路径），body/step 使用原训练池路径加 `#pool_id`，避免把增强数据误作独立来源。准备后已仅更正此身份元数据和相关哈希；CPU 逐数组精确检查确认时间、位置和四元数未变，检查记录在 `consumer_check.json`。后续课程需要明确选择实际配比并运行学习和独立评估；本批准备结果本身不证明训练收益。

## E2 selected for continuous learning

E1 completed with reproducible position improvement and remaining orientation/support failures. `decision.json` selects the candidate100 checkpoint and a fixed reset sampling mixture near40%/body30%/step30%; config changes only training demonstrations and their group selection. The first continuous budget is1000 iterations,98,304,000 interactions/20,000 updates, using the existing transfer interface once and later same-config resume. No additional reward coefficient is changed. Actual group exposure, errors, episode outcomes and rewards are logged.

The initial MuJoCo train10 readout exited0: every case completed60s with no fall. Position RMSE near4=.0665–.0673m, body low/high/lateral=.1281/.0365/.0413m, step3=.0368/.0657/.0370m; orientation and support remain separate. This establishes the starting point on these training tasks, not independent generalization.

## First E2 attempt stopped; one feedback parameter selected

The original1000 request completed78 iterations0–77,7,667,712 transitions and1,560 updates, then stopped normally through the stop file (actual exit0). This saved checkpoint77 including the final completed update; it is not1000 completed. All numerical checks were finite, but28,058 falls and zero timeouts occurred. Ended episodes averaged4.27/3.93/3.84s for near/body/step. Frozen checkpoint50 independently failed low/high/lateral/far at2.38/1.62/2.06/2.32s. The high case has no post2s samples and correctly retains null post-transition metrics.

`first_attempt_result.json` preserves failures. KL drift was already large before the first falls; the existing adaptive optimizer could not reduce LR below1e-5. The selected next controlled setting exposes that lower bound and uses1e-7, with initialLR still1e-5. Reward structure, std, tasks, initializer and seed remain identical. High negative no-positive-foot-load cost is a competing hypothesis if reducing update amplitude does not help. This change is an experiment, not an assertion of causality or a reason to abandon continuous skill learning. See `decision_lr_floor.json`; checkpoint50 matches51 iterations per arm before any unequal-budget extension.

## Matched checkpoint50: feedback recovery supported

`matched_checkpoint50.json` verifies only the effective LR floor differs and each snapshot follows51 actual iterations,5,013,504 transitions/1,020 optimizer updates. The lower-floor MuJoCo evaluation exited0 and completed all4 requested cases without falls; control failed all4. Lower-floor low/high/lateral/far position RMSE=.09769/.06933/.04362/.05663m; orientation=.38453/.32720/.21787/.32262rad. This preserves most initializer task performance and supports update amplitude as a collapse contributor. It does not establish purposeful support or full pose success. Continuous training remains active; no new reward parameter is selected.

The checkpoint50 support readout also exited0. Lower-floor low/high/lateral held-task elevated feet remain close to the initializer (low FL mean sphere-bottom height~.297m, high RR~.460m, lateral FL~.205m). No claim of support recovery follows from restored survival. `read_workspace_support.py` reuses the earlier verified FK method; E1 self-check output matches all four original cases exactly.

Optional hip-height CPU readout (`lr_floor_checkpoint50_hip_support.json`) prevents overinterpreting absolute clearance: during high hold the RR sphere bottom mean is.460m, but same-side thigh origin is.509m; bottom is never above that origin. Low FL center is consistently~.024m above thigh while its sphere bottom stays below. Zero crossing is a geometry descriptor, not a chosen reward margin. No above-hip penalty has been selected.

## Checkpoint250 during uninterrupted learning

MuJoCo workspace4 and CPU geometry readout both exited0. All requested durations completed without falls. Versus checkpoint50, low position/angle RMSE improves .09769/.38453→.08550/.27738; high .06933/.32720→.06558/.31401; lateral .04362/.21787→.04948/.23770; far .05663/.32262→.06296/.28761. Low/high improve while other metrics are mixed; no all-cases-monotonic rule is imposed. During high hold FL now rests near ground, but RR remains~.486m raised; low/lateral FL mean clearance .271/.153m remains prolonged. The selected continuous run is maintained. Compact development trajectory in `learning_readouts.json`; this is additional budget along one method, not a matched algorithm comparison.

## Checkpoint500: mixed progress and a retained failure

Workspace4 MuJoCo and geometry readout exited0; all four cases completed without falls, but whole-trace position worsened from checkpoint250 in all four (.10450/.07938/.05457/.07522m), while orientation was mixed (.35766/.37856/.22964/.26474rad). Far hold now has three feet near ground and FL~.129m clearance. High changed configuration: FL~.675m above ground and its sphere bottom~.246m above the corresponding thigh for99.5% of hold samples (`lr_floor_checkpoint500_hip_support.json`). This differs from checkpoint50 and is a real raised-above-hip configuration, not only body lifting. No reward was changed during this continuous run.

The separate train10 MuJoCo readout also exited0: nine cases completed60s, but body_lateral fell from low base height at8.94s before its20s final hold begins. Body_low whole/hold position improves .1281/.1349→.0669/.0599m; near4 position worsens to.0859–.0965m while angle RMSE improves to.2112–.2823rad. These are training tasks, not generalization. The failed hold has no samples and remains null in `train10_learning_readouts.json`. Continue the selected1000 budget with these failures retained rather than treating one checkpoint as a route verdict.


## Completed1000: task tradeoffs and support failure remain

The lower-floor continuous run completed all1000 iterations0–999 and exited0:98,304,000 interactions,20,000 updates,3593.926s training loop,1750 falls and32,874 resets; all numerical checks passed. Actual near/body/step exposure was40.564%/30.524%/28.912%. This budget follows one uninterrupted learning trajectory after the controlled LR repair; the first78-iteration failed attempt remains separate.

All six final evaluation commands exited0. PhysX workspace4 completed4/4; MuJoCo high fell at10.46s and the other3 completed. Both test8 suites completed8/8, but position worsened in every case versus the E1candidate initializer: mean PhysX.066775→.089112m and MuJoCo.067972→.099952m; orientation means improved. Training-task readout completed9/10 in each engine, with different failures: PhysX step01 at11.46s, MuJoCo step00 at16.18s. MuJoCo body_lateral recovered from the500-checkpoint failure and step02 improved, while body_low and other tasks regressed. `final1000_summary.json`, `learning_readouts.json`, and `train10_learning_readouts.json` retain original duration/failures and separate common-time comparisons from whole traces. No formal success follows.

Saved-state geometry confirms an actual training-distribution blind spot in the existing planar feet-under-hips reward. In PhysX, body_high/step00/step02 FL sphere bottoms are above the same-side thigh origin for84.34%/83.55%/81.83% of post2 samples, with median excess.260/.262/.277m. MuJoCo also shows sustained upward folding, but low differs between engines. Training step cases do move the base and change foot positions; these motions cannot be called stable gait given prolonged raised feet and failures. See `train1000_physx_training_support.json` and `train1000_training_support.json`. A single one-sided geometry cost is selected for the next controlled probe, not an automatic2000-iteration continuation or a claim that folding alone caused every fall.

Saved PhysX workspace low/far replays were rendered through existing MuJoCo geometry without new physics or policy execution. `video_replay/low.mp4` and `far.mp4` contain999/2999 frames at50Hz (19.98/59.98s); the last.02s lacks a following observation for aligned joints and is explicitly omitted. Both ffprobe frame counts and full ffmpeg decodes exited0; frame399 and each final aligned frame were visually checked. These are saved-state explanatory videos, not visual-control experiments. Renderer source and trace/spec identities are in their JSON sidecars.
