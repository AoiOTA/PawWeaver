# PawWeaver · 四足机械臂全身协同

目标：世界系末端位姿轨迹（位置和朝向）→ 单一 18 关节强化学习 Actor → 12 腿＋6 臂统一关节位置目标 → 显式 PD。不输入底盘速度指令，策略自主协调腿、臂和底盘运动；参考 MLM，使用因果目标历史及可选的预测未来目标，面向后续手持 UMI 遥控。Isaac Lab/PhysX 训练，独立 MuJoCo 验证。当前选用 **RealSense D435** 和松灵 Piper-H 腕部支架；原 DC1 配置保留在 `configs/cameras/dabai_dc1.json`。

## 当前状态

**当前正在执行（2026-09-10）**：[UMI-inspired 训练方案](artifacts/runs/diagnostic_pose_learning/umi_recipe/README.md) 已启动一个固定1000轮训练进程，随后进行双引擎 test8／far60 四项后测。采用全新站立初始化、奖励课程与更充分的腿部探索，检验整套方案的任务表现；不是单变量消融或已取得学习收益。当前原验收及 `trained=false` 边界不变。

世界末端位姿接口已实现：**276 维观测、单个 18 输出 Actor、schema-2 位姿轨迹与策略包**，贯通数据转换、因果位姿历史、Isaac/MuJoCo和视觉测量。1000轮工程训练已完成，独立测试显示位置改善、朝向退步；后续联合位姿奖励250轮及双引擎评估也已完成，平均位置／朝向RMSE为PhysX **3.85 cm／.129 rad**、MuJoCo **3.73 cm／.160 rad**。这仍不代表完整位姿任务或正式硬件验收成功，包为 `trained=false`，原验收数值保留，朝向验收阈值尚未确定。

腿部初始std×3与控制组的等预算250轮及双引擎test8均已完成：PhysX朝向均值改善，但MuJoCo八例朝向全退步，当前不支持采用std×3，见 [对照记录](artifacts/runs/diagnostic_pose_learning/leg_std_comparison/README.md)。原coupled250的MuJoCo远目标60秒测试在22.94秒跌倒；std远目标补测及后续stage2＋60秒训练250轮和四项后测均已完成。新策略两引擎远目标均存活60秒，但位置RMSE仍约62.1 cm，尚未学会远目标跟踪；位置奖励宽度.45候选也已完成对照，位置多例退步且MuJoCo远目标8.24秒跌倒，不采用.45，默认.15未改。腿均值越界正则.001候选的250轮训练和四项后测已完成：两引擎远目标均存活60秒，但位置、朝向RMSE均退步，不采用该候选，默认系数0及位置奖励宽度.15未改。冻结±5 cm目标回放显示越界幅度降低，却未恢复远状态髋关节和后腿持续响应；这是条件响应证据，不是动态因果证明，见 [实验与回放记录](artifacts/runs/diagnostic_pose_learning/leg_mean_bound_comparison/README.md)。

同起点progress权重1→10的250轮训练和四项后测已完成：test8共同窗内位置多数改善、朝向多数退步，PhysX一例16.44秒跌倒，MuJoCo八例均完成20秒。far共同窗内位置、朝向误差均改善，但PhysX／MuJoCo分别26.62／33.62秒跌倒，控制均完成60秒，当前候选不采用；不能把候选短程与控制全程均值比较为进步。默认progress1、bound0、width.15未改，见 [progress对照记录](artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/README.md)。随后同配置`--resume`追加750轮至累计1000轮和四项后测均已退出0，但不采用：test8两引擎均全20秒无跌倒，PhysX位置改善，而local4朝向双引擎全退步；far跌倒时间进一步缩短为PhysX26.62→10.90秒、MuJoCo33.62→12.70秒，共同窗位置／朝向RMSE均变差。[失稳回读](artifacts/runs/diagnostic_pose_learning/progress_weight_comparison/continuation_failure_analysis/README.md) 识别到倾斜触发终止和PhysX后足力下降，这些是过程观察，不能归因为奖励系数。短时冻结采集已完成，2.4秒窗口记录12次早期自然跌倒，属于startup/reset训练分布证据，不能当作far失稳重现。termination−5→−50的250轮训练与四项后测均已退出0，但不采用：test8两引擎全20秒无跌倒，位姿误差仍混合；PhysX far恢复60秒存活但位置RMSE仍.5958 m，MuJoCo在22.46秒更早跌倒。主同预算对照与稳定背景按各自共同窗保留，见 [终止权重对照](artifacts/runs/diagnostic_pose_learning/termination_weight_comparison/README.md)。[已保存数据的只读判别](artifacts/runs/diagnostic_pose_learning/termination_weight_comparison/next_learning_readout/README.md) 已完成CPU回放：termination−50未恢复持续髋／后腿有效目标响应。随后同起点、同预算P10/T50的 [softsign腿均值对照](artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/README.md) 250轮训练与四项后测均已退出0，**不采用**：test8两引擎均全20秒无跌倒，但PhysX位置／朝向各1好7差；PhysX far虽完整60秒，位置RMSE退步至.619860 m。MuJoCo far由22.46延至28.72秒仍跌倒，共同2–22.46秒位置／朝向RMSE由.350794→.384550 m／.114260→.187805 rad均退步。仅加入softsign映射，零更新时也改变腿PD目标；默认identity与原验收保留，opt-in代码保留复现。[学习后只读回放](artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/post_learning_readout/README.md) 与 [动作能力证据核对](artifacts/runs/diagnostic_pose_learning/softsign_mean_comparison/actuation_evidence/README.md) 已完成：softsign均值路径保留，保存远状态上髋关节高斯采样落入动作区间的解析概率约48%，但±5 cm目标扰动下RR thigh的PD目标响应仅约.00024 rad，未证明有效协调；现有加载证据支持站立／展开臂，尚无受控抬足证据。固定RR腿、当前动作范围的 [7秒双引擎同输入诊断](artifacts/runs/diagnostic_loaded_leg_lift/README.md) 已完成：两项有效运行均退出0、全7秒有限且无跌倒／饱和，但RR仅短暂卸载，FL反而离地，机身倾斜约19°且回程未恢复，未实现干净受控抬放。MuJoCo直接记录RR大腿接地，PhysX仅提供相应大腿净力证据；不能由这一固定输入证明动作范围不足。前两次诊断脚本失败及修复均保留，该诊断未改默认参数，softsign仍不采用。

[adaptive sampling对照](artifacts/runs/diagnostic_pose_learning/adaptive_sampling_comparison/README.md) 的250轮训练与四项固定后测均已实际退出0，**不采用候选、不追加预算或参数扫描**。唯一配置差异为`adaptive_sampling=false→true`，复用uniform softsign250同初始化、同预算控制；采样路径不同，不宣称逐轨迹配对。现有checkpoint确认概率与完成回合分布改变，采样并非未生效，但位置／跌倒EMA不含朝向／碰撞，不能替代位姿收益。test8两组两引擎均完整8×20秒无跌倒；共同2–20秒内PhysX朝向8例全退步，MuJoCo位置8例、朝向7例退步。PhysX far两组均60秒，位置RMSE仅.619860→.614141 m；MuJoCo由28.72秒跌倒变为完整60秒存活，但共同2–28.72秒位置RMSE **.572716→.598932 m**变差，候选完整60秒仍 **.621898 m**。存活改善不等于远目标跟踪成功，也不证明动作范围不足。默认配置、原验收与`trained=false`保持不变。

[新腿动作范围对照](artifacts/runs/diagnostic_pose_learning/leg_range_comparison/README.md) 的新.2控制与新.5候选各250轮及四项固定后测，**十条命令均实际退出0，不采用.5，不自动追加预算或尺度扫描**。每组24,576,000 transitions／5,000次更新，全部有限；训练跌倒1,549→2,362，子步力矩饱和0.31045%→0.55579%。两组两引擎test8均完整20秒无跌倒，共同2–20秒平均位置／朝向RMSE：PhysX **.027977→.036483 m／.100327→.188344 rad**，MuJoCo **.026513→.031052 m／.107312→.161160 rad**；朝向各8例全退步。MuJoCo位置单项通过数7→8保留，不能称完整位姿成功。far两组两引擎均完整60秒无跌倒，共同2–60秒PhysX位置仅.636364→.628724 m小幅改善、朝向.180363→.186254 rad变差；MuJoCo位置／朝向 **.621293→.638153 m／.148586→.242929 rad**均变差。逐例P95、时长与全部训练事件见实验记录。

初始确定性腿PD目标及裁剪前角度std匹配，不代表加载站姿；动作范围、裁剪／探索、归一化动作历史、动作变化率奖励及优化坐标属于联合处理，历史T50／softsign未替代新控制。[保存轨迹回读](artifacts/runs/diagnostic_pose_learning/leg_range_comparison/candidate_motion_readout/README.md) 显示候选两引擎2–30秒基座前进更少、最低高度更低；PhysX仅有50Hz RR足净力卸载记录，不能据此称有效步态或证明动作范围不足。[参考训练核对](artifacts/runs/diagnostic_pose_learning/leg_range_comparison/reference_training_audit/README.md) 已完成：UMI官方默认支持无步行检查点的单18动作直接训练，但目标表达、裁剪／探索、课程及PPO配方不同；没有因此选定新训练机制或预算。.5仍是临时工程参数，默认、原验收与`trained=false`不变。纯位姿学习稳定前暂缓视觉扩展；其余证据见 [运行手册](docs/runbook.md)，旧246维位置结果保留历史身份。

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
