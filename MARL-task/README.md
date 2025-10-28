# MATD3 for Berth Allocation with Shore Power Coordination

多智能体强化学习（MATD3）在连续泊位分配与岸电协同优化中的应用。

## 项目简介

本项目实现了基于MATD3（Multi-Agent Twin Delayed Deep Deterministic Policy Gradient）算法的自动化集装箱码头泊位分配与岸电协同优化系统。

### 核心特性

- **POMDP建模**: 部分可观测马尔可夫决策过程
- **CTDE架构**: 集中训练、分散执行
- **连续动作空间**: 泊位位置、等待时间、岸电使用概率
- **多目标优化**: 泊位利用率、船舶等待时间、碳排放
- **改进设计**: 基于论文评审意见的多项改进

### 论文对应改进

| 评审问题 | 改进实现 |
|---------|---------|
| 状态噪声混淆探索概念 | 17维观测（移除噪声维度），仅在动作层加噪声 |
| 分散靠泊奖励不合理 | 基于邻近船舶拥挤度的凸惩罚设计 |
| 船舶生成过于简单 | 非齐次泊松过程 + 多峰分布 |
| 缺少奖励权重消融 | 提供完整的敏感性分析框架 |
| 可视化不足 | 岸电负载曲线、泊位热力图、冲突检测 |

## 项目结构

```
MARL-task/
├── config/                      # 配置文件
│   └── default_config.yaml     # 默认配置
├── environment/                 # 环境模块
│   ├── vessel.py               # 船舶类与生成器
│   ├── shore_power.py          # 岸电管理器
│   └── berth_env.py            # 泊位环境
├── agents/                      # 智能体模块
│   ├── networks.py             # 神经网络（Actor/Critic）
│   ├── replay_buffer.py        # 经验回放池
│   └── matd3.py                # MATD3算法
├── rewards/                     # 奖励函数
│   └── reward_calculator.py    # 奖励计算器
├── training/                    # 训练模块
│   ├── trainer.py              # 训练器
│   └── evaluator.py            # 评估器
├── visualization/               # 可视化（待实现）
├── data/                        # 数据目录
├── results/                     # 结果目录
│   ├── models/                 # 模型保存
│   ├── logs/                   # 日志
│   └── figures/                # 图表
├── main.py                      # 主入口
├── requirements.txt             # 依赖
└── README.md                    # 本文档
```

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0

### 安装步骤

1. 克隆仓库（或使用本地目录）:
```bash
cd /Users/duan/mac-miclsirr/Sensors-models/MARL-task
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

使用默认配置训练MATD3智能体：

```bash
python main.py --mode train --config config/default_config.yaml
```

指定设备和随机种子：

```bash
python main.py --mode train --config config/default_config.yaml \
  --device cuda --seed 42
```

### 评估

评估训练好的模型：

```bash
python main.py --mode eval --model results/models/best_model.pth \
  --config config/default_config.yaml
```

### 配置文件

编辑 `config/default_config.yaml` 以修改：

- 环境参数（泊位长度、船舶数量等）
- 奖励权重（c1-c8）
- 网络结构
- 训练超参数

## 📚 完整文档

查看详细文档请访问：**[docs/README.md](docs/README.md)**

核心文档：
- **[RLlib完整指南](docs/RLLIB_COMPLETE_GUIDE.md)** - RLlib框架、环境实现、训练配置、并行优化、分布式训练
- **[算法设计](docs/ALGORITHM_DESIGN.md)** - 问题建模、环境设计、奖励函数、算法实现、改进优化
- **[系统架构](docs/SYSTEM_ARCHITECTURE.md)** - 整体架构、核心模块、数据流程、部署架构、开发指南

## 核心模块说明

### 1. 环境模块 (environment/)

#### 泊位环境 (BerthAllocationEnv)

- **观测空间**: 17维局部特征
  - 静态: 船长、到港时间、优先级、岸电能力
  - 动态: 当前时间、已等待时间、操作时间
  - 岸电: 5段使用率 + 总使用率
  - 泊位: 左右邻近距离、可用空间

- **动作空间**: 3维连续动作 [0, 1]
  - 位置: 靠泊位置（归一化）
  - 时间: 等待时长（归一化）
  - 概率: 岸电使用概率

#### 船舶生成器 (VesselGenerator)

支持两种模式：
- **realistic**: 非齐次泊松到港 + 多峰船长分布
- **simple**: 简单均匀分布（基线）

### 2. 智能体模块 (agents/)

#### MATD3算法

- **Actor**: 每个智能体独立策略网络
- **Critic**: 共享双Critic网络（集中价值估计）
- **关键机制**:
  - 双Q学习抑制高估
  - 延迟策略更新
  - 目标策略平滑
  - 分维度探索噪声

### 3. 奖励函数 (rewards/)

多目标奖励设计：

```python
reward = c1 * base_reward              # 基础正奖励
       - c2 * waiting_penalty          # 等待惩罚
       - c3 * emission_penalty         # 碳排放惩罚
       + c4 * shore_power_bonus        # 岸电奖励
       + c5 * utilization_reward       # 利用率奖励
       + c6 * spacing_reward           # 分散靠泊奖励（改进）
```

**改进点**: spacing_reward 基于邻近拥挤度，而非岸线中心距离

## 实验配置

### 默认参数

| 参数 | 值 | 说明 |
|------|-----|-----|
| 泊位长度 | 2000m | 连续岸线长度 |
| 规划周期 | 7天 | 规划时间窗 |
| 最大船舶数 | 20 | 每回合船舶数 |
| Actor学习率 | 1e-4 | 策略网络学习率 |
| Critic学习率 | 1e-3 | 价值网络学习率 |
| 折扣因子 | 0.99 | γ |
| 软更新系数 | 0.005 | τ |
| 策略延迟 | 2 | 延迟更新步数 |

### 训练建议

1. **初始训练**: 5000 episodes，评估间隔 100
2. **批量大小**: 256（根据显存调整）
3. **经验池**: 1M transitions
4. **设备**: 建议使用GPU加速

## 评估指标

- **berth_utilization**: 泊位利用率（时空占用率）
- **avg_waiting_time**: 平均等待时间（小时）
- **total_emissions**: 总碳排放（kg CO2）
- **shore_power_usage_rate**: 岸电使用率
- **avg_inference_time**: 平均推理时间（秒）

## 技术细节

### CTDE架构

```
训练时:
  Actor_i(obs_i) -> action_i
  Critic(global_state, all_actions) -> Q-value

执行时:
  Actor_i(obs_i) -> action_i  (分散执行)
```

### 双Critic机制

```python
target_q1 = Critic1_target(s', a')
target_q2 = Critic2_target(s', a')
target_q = min(target_q1, target_q2)  # 抑制高估
```

### 探索噪声设计

```python
# 位置: 高斯噪声
position += N(0, σ_pos)

# 时间: 高斯噪声
time += N(0, σ_time)

# 岸电概率: 均匀噪声
prob += U(-σ_prob, σ_prob)
```

## 故障排除

遇到问题？查看 **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** 获取解决方案。

常见问题：
- 启动失败 - 依赖未安装
- 端口被占用
- Conda环境问题
- TensorBoard无数据
- Web界面无法连接

## 常见任务

### 训练模型

```bash
python main.py --mode train --config config/default_config.yaml
```

### 加载真实数据

```python
from environment import VesselGenerator

generator = VesselGenerator(config)
vessels = generator.load_from_csv('data/real_world/vessels.csv')
observations, _ = env.reset(options={'vessels': vessels})
```

### 修改配置

编辑 `config/default_config.yaml` 调整参数

## 引用

如果使用本项目，请引用原始论文：

```bibtex
@article{your_paper_2025,
  title={Multi-Agent Reinforcement Learning for Berth Allocation
         with Shore Power Coordination},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## 许可证

MIT License

## 联系方式

- 问题反馈: [GitHub Issues](链接)
- 邮箱: your.email@example.com

## 快速开始

### 使用管理脚本（推荐）

```bash
# 查看帮助
./manage.sh help

# 检查系统环境和依赖
./manage.sh check

# 快速测试环境
./manage.sh test

# 一键启动所有服务
./manage.sh start
```

脚本会自动：
- ✅ 检查系统环境和Python依赖
- ✅ 启动后端 (Port 8000)
- ✅ 启动前端 (Port 3000)
- ✅ 启动TensorBoard (Port 6006)

### 管理命令

```bash
# 项目管理
./manage.sh check          # 检查系统环境和依赖
./manage.sh test           # 快速测试环境
./manage.sh archive        # 归档旧代码 (EPyMARL/MATD3)
./manage.sh clean          # 清理日志和临时文件

# 训练管理
./manage.sh train          # 启动RLlib训练
./manage.sh tensorboard    # 启动TensorBoard监控

# 服务管理
./manage.sh backend        # 启动后端服务 (Port 8000)
./manage.sh frontend       # 启动前端服务 (Port 3000)
./manage.sh start          # 启动所有服务
./manage.sh stop           # 停止所有服务

# 日志管理
./manage.sh logs training  # 查看训练日志
./manage.sh logs backend   # 查看后端日志
./manage.sh logs testing   # 查看测试日志
```

### 访问系统

启动后访问：
- **Web界面**: http://localhost:3000/index.html
- **API文档**: http://localhost:8000/docs
- **TensorBoard**: http://localhost:6006

### Web界面使用

1. 配置任务参数（船只数量、计划窗口等）
2. 点击"Generate Task"生成任务
3. 选择算法（MATD3/Greedy/FCFS）
4. 点击"Run Algorithm"运行
5. 查看可视化结果和性能指标

## 更新日志

### v2.1 (2025-10-28)

- ✅ **统一管理脚本** - manage.sh替代11个独立脚本
- ✅ **RLlib框架** - 完整迁移到RLlib 2.50.1
- ✅ **文档整合** - 3个核心文档（RLlib/算法/架构）
- ✅ **归档旧框架** - EPyMARL和MATD3已归档
- ✅ **自动资源优化** - rllib_train_advanced.py

### v2.0 (2025-01-26)

- ✅ **Web可视化系统** - 前后端完整实现
- ✅ **TensorBoard日志** - 训练实时可视化
- ✅ **Conda环境支持** - 自动环境管理
- ✅ **结构化日志** - JSON + 文本双重日志

### v1.0 (2025-10-25)

- ✅ 完整MATD3实现
- ✅ 连续泊位与岸电协同环境
- ✅ 改进的奖励函数（基于评审意见）
- ✅ 真实船舶生成（非齐次泊松）
- ✅ 训练与评估框架

## 致谢

感谢论文评审专家的宝贵意见和建议。
