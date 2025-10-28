# EPyMARL集成方案 - 泊位分配问题

## 📋 项目概述

将现有的MATD3泊位分配系统迁移到EPyMARL框架，利用其成熟的MARL基础设施、多种算法支持和规范化的实验流程。

---

## 🏗️ EPyMARL框架分析

### 核心组件

1. **环境接口** (`src/envs/`)
   - `multiagentenv.py`: 多智能体环境基类
   - `gymma.py`: Gymnasium环境包装器
   - 支持：共同奖励/个体奖励、部分可观测

2. **学习器** (`src/learners/`)
   - `maddpg_learner.py`: MADDPG (最接近MATD3)
   - `ppo_learner.py`: PPO系列
   - `actor_critic_learner.py`: Actor-Critic基类

3. **控制器** (`src/controllers/`)
   - 管理智能体策略选择
   - 支持参数共享/不共享

4. **模块** (`src/modules/`)
   - Actor/Critic网络定义
   - 支持RNN、MLP等架构

5. **运行器** (`src/runners/`)
   - 训练循环管理
   - 数据收集与批处理

---

## 📐 实施方案

### 阶段1: 环境适配 (Week 1-2)

#### 1.1 创建泊位分配环境包装器

**文件**: `epymarl/src/envs/berth_allocation_wrapper.py`

```python
from envs.multiagentenv import MultiAgentEnv
import gymnasium as gym
import numpy as np
from typing import Dict, List

class BerthAllocationEnv(MultiAgentEnv):
    """
    泊位分配环境 - EPyMARL适配

    特性:
    - 部分可观测（POMDP）
    - 连续动作空间 -> 离散化或保持连续
    - 个体奖励（每个船舶独立奖励）
    - 动态智能体数量（每个episode船舶数可变）
    """

    def __init__(self, **kwargs):
        # 从现有environment/berth_env.py适配
        self.berth_length = kwargs.get('berth_length', 2000)
        self.max_vessels = kwargs.get('max_vessels', 20)
        self.planning_horizon = kwargs.get('planning_horizon', 168)  # 7天

        # EPyMARL要求的属性
        self.n_agents = self.max_vessels
        self.n_actions = self._get_action_space_size()
        self.episode_limit = self.max_vessels  # 每艘船一个决策步

        # 观测空间 (17维局部观测)
        self.obs_shape = 17

        # 状态空间 (全局状态，用于集中式训练)
        self.state_shape = self._get_state_size()

    def reset(self):
        """重置环境，生成新的船舶任务"""
        # 从vessel.py的VesselGenerator生成船舶
        # 返回: obs, state
        pass

    def step(self, actions):
        """执行动作，返回奖励和下一状态

        Args:
            actions: [n_agents] 动作数组

        Returns:
            reward: float (common) 或 [n_agents] (individual)
            terminated: bool
            info: dict
        """
        pass

    def get_obs(self):
        """返回所有智能体的观测 [n_agents, obs_shape]"""
        pass

    def get_obs_agent(self, agent_id):
        """返回单个智能体的观测 [obs_shape]"""
        pass

    def get_state(self):
        """返回全局状态 [state_shape]"""
        pass

    def get_avail_actions(self):
        """返回每个智能体的可用动作掩码"""
        pass

    def get_env_info(self):
        """返回环境信息字典"""
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
```

#### 1.2 动作空间设计

**选项A: 离散化（推荐用于QMIX/VDN）**
```python
# 位置: 20个离散位置 (0, 100, 200, ..., 1900)
# 时间: 10个等待时段 (0h, 4h, 8h, ..., 36h)
# 岸电: 2个选项 (使用/不使用)
# 总动作数: 20 × 10 × 2 = 400
```

**选项B: 连续动作（推荐用于MADDPG/MATD3）**
```python
# 保持原有3维连续动作 [position, wait_time, shore_power_prob]
# 需要修改EPyMARL以支持连续动作空间
```

#### 1.3 观测空间设计

```python
def _get_observation(self, vessel_id):
    """
    17维局部观测（与原MATD3一致）:

    静态特征 (4):
    - vessel.length / berth_length
    - vessel.arrival_time / planning_horizon
    - vessel.priority / 4
    - vessel.can_use_shore_power

    动态特征 (3):
    - current_time / planning_horizon
    - waiting_time / max_wait_time
    - vessel.operation_time / planning_horizon

    岸电信息 (6):
    - shore_power_segment_usage (5 segments)
    - total_shore_power_usage

    泊位信息 (4):
    - left_neighbor_distance
    - right_neighbor_distance
    - available_space_ratio
    - current_berth_occupancy
    """
    return np.array([...])  # shape: (17,)
```

#### 1.4 状态空间设计（集中式训练用）

```python
def get_state(self):
    """
    全局状态（所有智能体可见）:

    - 当前时间
    - 所有船舶的静态特征矩阵 [max_vessels, 4]
    - 所有船舶的等待时间 [max_vessels]
    - 泊位占用情况（网格化） [grid_size]
    - 岸电段使用情况 [5]
    - 已分配船舶掩码 [max_vessels]

    展平后总维度: ~200-300
    """
    return np.concatenate([...])
```

---

### 阶段2: 算法实现 (Week 3-4)

#### 2.1 创建MATD3学习器

**文件**: `epymarl/src/learners/matd3_learner.py`

基于`maddpg_learner.py`扩展，添加:
- ✅ Twin Delayed Q-learning (双Critic)
- ✅ Delayed Policy Updates (延迟策略更新)
- ✅ Target Policy Smoothing (目标平滑)
- ✅ 连续动作空间支持

```python
class MATD3Learner:
    """
    Multi-Agent Twin Delayed DDPG Learner

    改进点（相比MADDPG）:
    1. 双Critic网络减少Q值高估
    2. 延迟策略更新稳定训练
    3. 目标策略平滑抑制noise
    """

    def __init__(self, mac, scheme, logger, args):
        # 创建两个Critic网络
        self.critic_1 = critic_registry[args.critic_type](scheme, args)
        self.critic_2 = critic_registry[args.critic_type](scheme, args)

        # 对应的目标网络
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # 策略延迟计数器
        self.policy_delay = args.policy_delay  # 默认2
        self.updates = 0

    def train(self, batch, t_env, episode_num):
        # Critic更新（每次都更新）
        self._train_critic(batch)

        # Actor更新（每policy_delay次更新一次）
        if self.updates % self.policy_delay == 0:
            self._train_actor(batch)
            self._update_targets()  # 软更新

        self.updates += 1
```

#### 2.2 创建连续动作控制器

**文件**: `epymarl/src/controllers/matd3_controller.py`

```python
class MATD3Controller:
    """支持连续动作的控制器"""

    def select_actions(self, ep_batch, t_ep, t_env,
                      bs=slice(None), test_mode=False):
        """
        选择连续动作

        Returns:
            actions: [batch_size, n_agents, action_dim]
        """
        # 获取Actor输出
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # 添加探索噪声（训练时）
        if not test_mode:
            agent_outputs = self._add_exploration_noise(agent_outputs)

        # Tanh激活保证在[-1, 1]范围
        actions = torch.tanh(agent_outputs)

        return actions

    def _add_exploration_noise(self, actions):
        """
        维度相关的探索噪声:
        - 位置维度: 高斯噪声 σ=0.1
        - 时间维度: 高斯噪声 σ=0.1
        - 岸电概率: 均匀噪声 ±0.1
        """
        pass
```

#### 2.3 网络架构

**文件**: `epymarl/src/modules/agents/matd3_agent.py`

```python
class MATD3Agent(nn.Module):
    """
    MATD3 Actor网络

    输入: observation (17维)
    输出: continuous actions (3维)
    """

    def __init__(self, input_shape, args):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)  # 256
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 3)  # 3维连续动作

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)  # 不加激活，后续用tanh
        return actions
```

**文件**: `epymarl/src/modules/critics/matd3_critic.py`

```python
class MATD3Critic(nn.Module):
    """
    集中式Critic网络

    输入:
    - states: 全局状态
    - actions: 所有智能体的动作 [n_agents * action_dim]

    输出: Q值 (标量)
    """

    def __init__(self, scheme, args):
        super().__init__()

        input_dim = (scheme["state"]["vshape"] +
                    args.n_agents * 3)  # state + all actions

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, args.n_agents)  # 每个agent一个Q值

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
```

---

### 阶段3: 配置与集成 (Week 5)

#### 3.1 算法配置文件

**文件**: `epymarl/src/config/algs/matd3.yaml`

```yaml
# MATD3算法配置
name: "matd3"

# 智能体参数
agent: "matd3_agent"
agent_output_type: "continuous"
action_selector: "continuous"

# Mac参数
mac: "matd3_mac"

# Critic参数
critic_type: "matd3_critic"
critic_lr: 0.001
target_update_mode: "soft"  # soft update
tau: 0.005

# Actor参数
lr: 0.0001
hidden_dim: 256

# MATD3特有参数
policy_delay: 2  # 延迟策略更新
policy_noise: 0.2  # 目标策略噪声
noise_clip: 0.5  # 噪声裁剪
exploration_noise:
  position: 0.1
  time: 0.1
  probability: 0.1

# 训练参数
batch_size: 256
buffer_size: 1000000
gamma: 0.99
grad_norm_clip: 10

# 奖励相关
standardise_rewards: True
standardise_returns: False
common_reward: False  # 使用个体奖励

# 其他
use_cuda: False
```

#### 3.2 环境配置文件

**文件**: `epymarl/src/config/envs/berth_allocation.yaml`

```yaml
env: berth_allocation

env_args:
  # 泊位参数
  berth_length: 2000
  max_vessels: 20
  planning_horizon: 168  # 7天 × 24小时

  # 船舶生成
  generation_mode: "realistic"  # realistic / simple

  # 岸电参数
  shore_power_enabled: True
  num_shore_power_segments: 5
  shore_power_capacity: 500  # kW per segment

  # 奖励权重
  reward_weights:
    c1_base: 100.0
    c2_waiting: 10.0
    c3_emission: 0.01
    c4_shore_power: 50.0
    c5_utilization: 200.0
    c6_spacing: 30.0

  # 环境约束
  min_spacing: 20.0  # 最小船舶间距
  max_wait_time: 48.0  # 最大等待时间（小时）

t_max: 1000000  # 总训练步数
```

#### 3.3 注册环境

**文件**: `epymarl/src/envs/__init__.py`

```python
from envs.berth_allocation_wrapper import BerthAllocationEnv

REGISTRY["berth_allocation"] = BerthAllocationEnv
```

---

### 阶段4: 后端集成 (Week 6)

#### 4.1 修改后端算法运行器

**文件**: `backend/services/epymarl_runner.py`

```python
import sys
sys.path.append('epymarl/src')

from run import run as epymarl_run
from types import SimpleNamespace

class EPyMARLRunner:
    """EPyMARL算法运行器"""

    def __init__(self):
        self.logger = setup_logger("epymarl")

    def run_algorithm(self, task_data, algorithm, model_path=None):
        """
        运行EPyMARL算法

        Args:
            task_data: 任务数据（包含vessels和config）
            algorithm: 算法名称 (matd3, maddpg, qmix等)
            model_path: 预训练模型路径

        Returns:
            allocations: 分配结果
            metrics: 性能指标
        """

        # 构建EPyMARL参数
        args = SimpleNamespace(
            config=algorithm,  # matd3
            env_config='berth_allocation',

            # 任务特定参数
            env_args={
                'vessels': task_data['vessels'],
                'berth_length': task_data['config']['berth_length'],
                # ... 其他参数
            },

            # 训练/评估模式
            evaluate=True,  # 评估模式（不训练）
            load_step=model_path,  # 加载模型

            # 日志
            use_tensorboard=False,
            use_wandb=False,
        )

        # 运行EPyMARL
        results = epymarl_run(args)

        # 提取结果
        allocations = self._extract_allocations(results)
        metrics = self._calculate_metrics(allocations)

        return {
            'allocations': allocations,
            'metrics': metrics
        }
```

#### 4.2 更新API端点

**文件**: `backend/api/algorithm.py`

```python
from backend.services.epymarl_runner import EPyMARLRunner

epymarl_runner = EPyMARLRunner()

@router.post("/run", response_model=AllocationResponse)
async def run_algorithm(request: AlgorithmRunRequest):
    # 检查是否使用EPyMARL算法
    epymarl_algorithms = ['MATD3', 'MADDPG', 'QMIX', 'MAPPO']

    if request.algorithm in epymarl_algorithms:
        # 使用EPyMARL运行器
        result = epymarl_runner.run_algorithm(
            task_data=task_manager.get_task(request.task_id),
            algorithm=request.algorithm.lower(),
            model_path=request.model_path
        )
    else:
        # 使用原有的algorithm_runner (Greedy, FCFS)
        result = algorithm_runner.run(...)

    return AllocationResponse(...)
```

---

### 阶段5: 训练脚本 (Week 7)

#### 5.1 创建训练脚本

**文件**: `train_epymarl.py`

```python
#!/usr/bin/env python3
"""
EPyMARL训练脚本 - 泊位分配问题
"""

import sys
sys.path.append('epymarl/src')

from main import main as epymarl_main

if __name__ == '__main__':
    # 训练MATD3
    args = [
        '--config=matd3',
        '--env-config=berth_allocation',
        'with',
        'env_args.max_vessels=20',
        'env_args.berth_length=2000',
        'common_reward=False',  # 个体奖励
        't_max=1000000',
        'test_interval=10000',
        'save_model=True',
        'save_model_interval=50000',
    ]

    epymarl_main(args)
```

#### 5.2 运行命令

```bash
# 训练MATD3（个体奖励）
python train_epymarl.py

# 训练QMIX（共同奖励）
python epymarl/src/main.py \
  --config=qmix \
  --env-config=berth_allocation \
  with common_reward=True

# 评估训练好的模型
python epymarl/src/main.py \
  --config=matd3 \
  --env-config=berth_allocation \
  evaluate=True \
  load_step=500000 \
  render=True
```

---

## 📊 对比：当前实现 vs EPyMARL

| 维度 | 当前实现 | EPyMARL实现 |
|------|---------|------------|
| **代码复用** | 自己实现所有组件 | 复用成熟框架 |
| **算法支持** | 仅MATD3 + 基线 | MATD3, MADDPG, QMIX, VDN, MAPPO等 |
| **实验管理** | 简单日志 | Sacred + TensorBoard + W&B |
| **超参数搜索** | 手动 | 内置grid search |
| **可复现性** | 中等 | 高（配置文件+种子管理） |
| **社区支持** | 无 | 活跃（论文引用>1000） |
| **扩展性** | 需要大量修改 | 只需添加环境wrapper |
| **训练稳定性** | 依赖自己调试 | 已验证的技巧 |

---

## 🎯 推荐实施路线

### 快速原型（2周）
1. ✅ 实现`BerthAllocationEnv`环境wrapper
2. ✅ 配置MADDPG（最接近现有MATD3）
3. ✅ 运行baseline实验验证环境正确性

### 完整迁移（6周）
1. Week 1-2: 环境适配
2. Week 3-4: MATD3学习器实现
3. Week 5: 配置与测试
4. Week 6: 后端集成
5. Week 7: 训练与调优

### 长期优化（持续）
1. 添加更多算法（QMIX、MAPPO）
2. 超参数搜索与ablation study
3. 与论文评审意见对应的实验
4. 生成论文图表

---

## 📦 依赖更新

**新增依赖** (`requirements.txt`):
```txt
# EPyMARL核心
sacred>=0.8.4
GitPython
tensorboard-logger>=0.1.0

# 可选
wandb  # Weights & Biases日志
```

---

## 🚀 优势总结

1. **学术认可**: EPyMARL是MARL领域标准框架，论文使用它更有说服力
2. **算法丰富**: 一次环境适配，免费获得8+种算法
3. **实验规范**: Sacred配置管理，实验可复现
4. **调试便利**: 成熟的日志和可视化工具
5. **持续维护**: 2024年仍在活跃更新
6. **论文友好**: 内置绘图脚本，直接生成论文图表

---

## ⚠️ 注意事项

1. **动作空间**: EPyMARL原生支持离散动作，连续动作需要扩展
2. **动态智能体**: 需要处理每个episode船舶数量变化
3. **部分可观测**: 确保POMDP设置正确（observation ≠ state）
4. **奖励设计**: 验证个体奖励vs共同奖励对不同算法的影响
5. **计算资源**: EPyMARL训练较慢，建议使用GPU

---

## 📚 参考资料

- EPyMARL GitHub: https://github.com/uoe-agents/epymarl
- Blog Post: https://agents.inf.ed.ac.uk/blog/epymarl/
- PyMARL Paper: https://arxiv.org/abs/1910.00091
- MADDPG Paper: https://arxiv.org/abs/1706.02275

---

## 📝 下一步行动

1. **确认方案**: 与您确认是否采用此方案
2. **创建分支**: `git checkout -b feature/epymarl-integration`
3. **开始实现**: 从环境wrapper开始
4. **边实现边测试**: 每个阶段验证功能

请告诉我您的意见，我可以立即开始实施！
