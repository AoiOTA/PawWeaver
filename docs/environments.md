# Conda 环境与本地版本管理

项目路径：`/home/lyb/pawweaver`。环境按职责分为三个：

| 环境 | 用途 |
|---|---|
| `pawweaver-train` | 从已有 `isaacsim` 克隆，增加锁定版本的 Isaac Lab、RSL-RL 和本项目 |
| `pawweaver-runtime` | 全新 Python 3.12 环境；CPU PyTorch、MuJoCo 与视觉工具；不安装 Isaac Sim/Isaac Lab |
| `pawweaver-data` | 全新 Python 3.12 环境；NumPy、HDF5、SciPy；不安装 PyTorch 或仿真器 |

```bash
python scripts/setup_environments.py --role all --isaac-env isaacsim
conda activate pawweaver-runtime
bash scripts/check.sh
pawweaver fetch-assets
pawweaver audit
```

安装命令仅指向以上三个环境，不向 `base`、系统 Python、原有 `isaacsim` 安装软件。旧项目位置的符号链接仅保持当前 Codex 任务可访问，不是第二份工作区。

## Isaac Lab 6.0.1 依赖元数据修正

固定标签 `v3.0.0-beta2.patch1` 内的包版本号与标签号不同，属上游的独立包版本规则。安装时发现两项互相矛盾的依赖声明：

- `isaacsim-core` 要求 `packaging==26.0`，`isaaclab_rl` 却要求 `<24`。
- `isaacsim-kernel` 要求 `coverage==7.4.4`，`isaaclab` 却要求 `7.6.1`。

`setup_environments.py` 对该固定标签作可重复的本地元数据修正，保留 Isaac Sim 要求的版本；不改物理引擎或训练算法。修正经过 `pip check` 和后续启动检查验证。升级标签前必须重新核对，脚本遇到不同提交会拒绝继续。

测试禁用第三方 pytest 插件自动加载，并清除继承的 ROS `PYTHONPATH`；这不修改用户 shell 或 ROS 安装。

依赖解析记录输出到 `artifacts/environments/`。本地功能提交使用 `codex/` 分支，保留 `main` 初始基线；不配置远程仓库、不推送。大模型和日志通过忽略规则排除，但下载版本与校验算法保存在 Git 中。
