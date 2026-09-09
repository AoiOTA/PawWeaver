# PawWeaver · 四足机械臂全身协同

目标：世界系末端位姿轨迹（位置和朝向）→ 单一 18 关节强化学习 Actor → 12 腿＋6 臂统一关节位置目标 → 显式 PD。不输入底盘速度指令，策略自主协调腿、臂和底盘运动；参考 MLM，使用因果目标历史及可选的预测未来目标，面向后续手持 UMI 遥控。Isaac Lab/PhysX 训练，独立 MuJoCo 验证。当前选用 **RealSense D435** 和松灵 Piper-H 腕部支架；原 DC1 配置保留在 `configs/cameras/dabai_dc1.json`。

## 当前状态

世界末端位姿接口已实现：**276 维观测、单个 18 输出 Actor、schema-2 位姿轨迹与策略包**，贯通数据转换、因果位姿历史、Isaac/MuJoCo和视觉测量。1000轮工程训练已完成，独立测试显示位置改善、朝向退步；后续联合位姿奖励250轮及双引擎评估也已完成，平均位置／朝向RMSE为PhysX **3.85 cm／.129 rad**、MuJoCo **3.73 cm／.160 rad**。这仍不代表完整位姿任务或正式硬件验收成功，包为 `trained=false`，原验收数值保留，朝向验收阈值尚未确定。

腿部初始std×3与控制组的等预算250轮及双引擎test8均已完成：PhysX朝向均值改善，但MuJoCo八例朝向全退步，当前不支持采用std×3，见 [对照记录](artifacts/runs/diagnostic_pose_learning/leg_std_comparison/README.md)。原coupled250的MuJoCo远目标60秒测试在22.94秒跌倒；std远目标补测及后续stage2＋60秒训练250轮和四项后测均已完成。新策略两引擎远目标均存活60秒，但位置RMSE仍约62.1 cm，尚未学会远目标跟踪；位置奖励宽度.45候选也已完成对照，位置多例退步且MuJoCo远目标8.24秒跌倒，不采用.45，默认.15未改。腿均值越界正则.001候选的250轮训练和四项后测已完成：两引擎远目标均存活60秒，但位置、朝向RMSE均退步，不采用该候选，默认系数0及位置奖励宽度.15未改。冻结±5 cm目标回放显示越界幅度降低，却未恢复远状态髋关节和后腿持续响应；这是条件响应证据，不是动态因果证明，见 [实验与回放记录](artifacts/runs/diagnostic_pose_learning/leg_mean_bound_comparison/README.md)。

同起点progress权重1→10的250轮训练和四项后测已完成：test8共同窗内位置多数改善、朝向多数退步，PhysX一例16.44秒跌倒，MuJoCo八例均完成20秒。far共同窗内位置、朝向误差均改善，但PhysX／MuJoCo分别26.62／33.62秒跌倒，控制均完成60秒，当前候选不采用；不能把候选短程与控制全程均值比较为进步。默认progress1、bound0、width.15未改，见 [progress对照记录](artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/README.md)。已选择保留progress10配置，从其candidate249经`--resume`追加750轮至该配置累计1000轮；续训准备／复核中，尚未启动。纯位姿学习稳定前暂缓视觉扩展；其余证据见 [运行手册](docs/runbook.md)，旧246维位置结果保留历史身份。

项目持续以真实 PawWeaver 工作 dogfood `/home/lyb/kiss-my-agent-dogfood`：以减少无用设计、避免阻塞和缩短研究循环为目标。已移除无消费用途的 bundle 重复 metadata，并让批量评估继续推进，不把旧精确重放或新增完整 bundle hash 比较变成任务门槛；实际变化及其证据限制见 [验证记录](docs/validation.md)。

组合机器人已有 URDF、USD 和 MJCF；机械臂贴合转接板，D435 与支架已装在腕部。AS2 按名称和坐标系核对后使用官方 MuJoCo 惯性参数，组合参考质量约 25.682 kg；转接板、导轨支撑和线缆尚未计入有效动力学，相机惯量属于诊断近似。详细证据及剩余事项见 [验证记录](docs/validation.md) 和 [硬件参数](docs/hardware.md)。

工作区位于 `/home/lyb/pawweaver`。软件分别安装在 `pawweaver-train`、`pawweaver-runtime` 和 `pawweaver-data` 三个 Conda 环境；不修改 `base` 或原有 `isaacsim`。`.deps/`、下载的模型、数据、日志、权重和生成产物不纳入 Git；版本与来源由配置及校验记录管理。

## 资产与检查

```bash
python scripts/setup_environments.py --role all
conda activate pawweaver-runtime
pawweaver fetch-assets
pawweaver audit
```

`audit` 将报告写入 `artifacts/audit/`；M0 未闭合时返回退出码 2。原始模型不会被修改。

环境安装与兼容性修正详见 [环境说明](docs/environments.md)。

运行、训练和评估命令见 [运行手册](docs/runbook.md)。依赖快照保存于 `environments/locks/`。预览为 `artifacts/preview/robot.png`，环绕视频为 `artifacts/preview/asset-orbit.mp4`。

## 设计边界

- AS2 EDU 12 腿关节 + Piper-H 6 臂关节，固定 40 mm 夹爪开度。
- 仅以世界系末端位置与朝向轨迹为任务命令，由同一个 Actor 自主协调全身；无外部底盘速度命令。基座速度仅作估计量或 Critic／训练标签。
- 策略周期 20 ms，物理步长 2 ms。
- 关节顺序：FR、FL、RR、RL 各 hip/thigh/calf，然后 arm_joint1…6。
- 四元数统一 WXYZ，所有长度和动力学量使用 SI 单位。
- D435 的 RGB/深度显式对齐；实际设备标定与虚拟针孔参数分开记录。
- 初版不包含抓取接触、真机控制或自主全局定位。

## 本地 Git

按功能阶段进行本地提交，并在每次提交前运行相应检查。当前不配置远程仓库，也不执行 push。后续新增远程不会影响已有本地历史。
