# PawWeaver · 四足机械臂全身协同

世界系末端目标 → 单一 18 关节 Actor → 显式 PD。Isaac Lab/PhysX 训练，独立 MuJoCo 验证。当前选用 **RealSense D435** 和松灵 Piper-H 腕部支架；原 DC1 配置保留在 `configs/cameras/dabai_dc1.json`。

## 当前状态

已实现资产生成、全身任务、PPO、速度估计/轨迹预测、自适应采样、FastUMI TCP 转换、策略导出、固定测试集、独立推理和视觉运行器。**还没有经过任务验收的 AS2 + Piper-H 策略，正式硬件组合训练尚未开始。** 小规模 PPO、执行器阶跃和视觉测试使用明确标识的合成机器人，只验证软件链路。

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
- 任务坐标系在 episode 初始化时固定；无外部底盘速度命令。
- 策略周期 20 ms，物理步长 2 ms。
- 关节顺序：FR、FL、RR、RL 各 hip/thigh/calf，然后 arm_joint1…6。
- 四元数统一 WXYZ，所有长度和动力学量使用 SI 单位。
- D435 的 RGB/深度显式对齐；实际设备标定与虚拟针孔参数分开记录。
- 初版不包含抓取接触、真机控制或自主全局定位。

## 本地 Git

按功能阶段进行本地提交，并在每次提交前运行相应检查。当前不配置远程仓库，也不执行 push。后续新增远程不会影响已有本地历史。
