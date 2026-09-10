# 参数来源与未闭合事项

用户允许在 DC1 资料不足时更换成熟 RGB-D 相机。当前选用 RealSense D435：松灵提供 Piper-H、夹爪、腕部支架与相机的装配模型，观测距离也比以近距操作为主的 D405 更符合第一版远距目标跟踪。

| 项目 | 找到的证据 | 使用方式与限制 |
|---|---|---|
| AS2 | 官方 MuJoCo 提交 `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`，20.7 kg；原 URDF 17.64 kg | 核对 link 坐标和关节轴后读取官方惯性，不按总重缩放；具体 EDU 实物配置仍待确认 |
| AS2 质量差异 | base 8.8→11.178 kg；四个 thigh 各 1.25→1.3525 kg；四个 calf 各 0.22→0.288 kg | 解释了 3.06 kg 差异 |
| Piper-H | 本体 URDF 4.167 kg，手册 4.5 kg；官方 2025-12-26 STEP 已下载 | STEP 有 21 个 solid，无质量/密度记录；没有推测内部器件密度补造缺失质量 |
| 机械臂坐标 | 100 个随机姿态的 SDK MDH 与 URDF 一致 | 几何零位有独立对照；硬件/固件限位版本仍需确定 |
| 夹爪 | 源模型惯量，固定开度 40 mm | TCP 为夹爪基座 z=0.138 m 的指端平面，仍需确认实际工具中心 |
| D435 | 厂商表 3-52：75 g，公差 ±10%，90×25×25 mm | 诊断模型采用标称质量与均匀盒体惯量估计；源 URDF 明确注明惯量不可靠，未照搬 |
| 腕部支架 | 松灵提供网格、装配位姿和 0.2 kg 参考质量 | 仍非实物测量 |
| 相机光学 | 名义视场和源 xacro 的 15 mm RGB/深度偏移；1280×720、30 Hz | 生成带来源的虚拟针孔相机，不冒充设备序列号标定 |
| 背部安装 | 双轨中心距 152 mm；依据壳体表面设置支撑与转接板 | 机械臂与板面贴合；板和支撑为设计概念，动力学未纳入 |
| 驱动 | AS2 官方仿真参数与 Piper-H SDK 命令范围已保存 | SDK 编码范围不等于连续电机能力；限矩、默认姿态、动作尺度和 PD 仍需确定与检查 |
| 线缆 | 尚无走线和质量分布 | 未纳入有效动力学 |

组合参考质量约 25.682 kg = AS2 20.7 + Piper-H 4.167 + 夹爪 0.54 + D435 0.075 + 腕部支架 0.2。它不是完整实物称重结果。

版本固定于 `configs/sources.json`，驱动参考在 `configs/actuator_references.json`，相机资料在 `configs/cameras/realsense_d435.json`。生成资产 manifest 包含近似说明和文件校验值。

M0 的六项 `verified_overrides` 仍需补齐带来源的实际参数：装配、相机及支架、线缆、质量差异、关节映射和执行器。相机更换没有自动消除整机动力学的待核实项；软件检查通过也不代表 M1–M4 成功率已达标。

## 真实控制接口尚未接入

当前统一18关节链路的实际终点是仿真：[MujocoRunner](../src/pawweaver/mujoco_runtime.py) 从 `mjData` 读取状态并将 [JointPD](../src/pawweaver/control.py) 结果写入 `data.ctrl`；现有 `src/`、`scripts/` 没有 AS2 EDU 低层或 Piper-H SDK 通信消费者。当前50Hz Actor动作经 `q_target=q0+action_scale*clip(action,-1,1)` 和位置限位后，由500Hz显式PD计算并限矩。`ActuatorSpec.velocity` 虽有字段，此控制函数没有按该字段限制目标变化率，不能据此称已具备实机速度保护。以上均未证明真实设备具有相同位置／MIT／力矩闭环、频率和响应。

接入前尚需按实际设备确认：

- AS2 EDU、Piper-H具体型号／固件及可用低层模式；18关节设备索引、方向、零位和适用限位。当前顺序为FR、FL、RR、RL各hip/thigh/calf，再arm_joint1…6，见[接口契约](../src/pawweaver/contracts.py)。
- 两设备指令与反馈频率、时间戳／延迟、底层增益、持续执行能力及控制中断后的设备行为；SDK编码范围不能替代这些信息。
- 同步关节状态、IMU与世界系基座位姿来源，结合安装和TCP标定计算世界TCP；当前观测历史使用基座位姿转换世界目标，仿真真值不能替代实机定位。

本地固定版本资料仍有待实物版本裁决的差异：[手册限位配置](../configs/hardware.json) 与 [SDK参考](../configs/actuator_references.json) 的Piper-H J2为195°／180°、J4为±127°／±135°、J6为±170°／±180°。此前MDH与URDF的100姿态一致只核对了几何，未关闭设备映射或限位版本差异。AS2增益／力矩来自官方仿真配置，Piper-H力矩范围来自CAN编码，均未证明实际持续能力。以上确认与临时参数仿真并行；本轮只核查了本地代码和固定资料，未连接、探测或操作设备。

## 官方资料

- [AS2 MuJoCo 模型](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/as2/as2.xml)
- [Piper-H + D435 装配](https://github.com/agilexrobotics/agx_arm_sim/blob/f8cd8b147c75d59e14f90fb0646770eefa268ed0/agx_arm_description/urdf/piper_h_gripper_d435.urdf)
- [Piper-H STEP 附件](https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/glzd7a853owrmsk0?singleDoc)
- [D435 产品资料](https://www.realsenseai.com/products/stereo-depth-camera-d435/)
- [D400 系列数据手册](https://www.realsenseai.com/download/21345/)
