# RLlib实施方案 - 泊位分配MARL项目

**框架**: Ray RLlib
**环境**: Mac本地开发 + 云GPU训练
**时间**: 3个月（2025年10月 - 2026年1月）
**算法**: 多智能体强化学习（MADDPG, MAPPO等）

---

## 🎯 项目目标

1. ✅ 使用RLlib实现泊位分配的多智能体环境
2. ✅ 训练MADDPG等MARL算法
3. ✅ 在云GPU上进行大规模实验
4. ✅ 完成学术论文（2026年1月前）

---

## 📅 时间规划（12周）

```
Week 1 (10月末):        RLlib环境开发（Mac本地）
Week 2 (11月初):        多智能体算法配置
Week 3 (11月中):        云平台部署准备
Week 4-6 (11月):        云端大规模实验
Week 7-10 (12月):       论文撰写
Week 11-12 (1月初):     修改润色投稿
```

---

## 🏗️ 阶段1: RLlib环境开发（Week 1）

### 1.1 安装依赖（Day 1）

#### Mac本地开发环境

```bash
# 1. 安装Ray和RLlib
conda activate marl-task
pip install "ray[rllib]">=2.9.0
pip install "gymnasium>=0.28.0"
pip install torch  # PyTorch框架

# 2. 验证安装
python -c "import ray; ray.init(); print('Ray version:', ray.__version__)"
python -c "from ray.rllib.algorithms.ppo import PPOConfig; print('RLlib OK')"

# 3. 安装可选依赖
pip install tensorboard
pip install wandb  # Weights & Biases日志（可选）
```

#### 云平台环境（后续准备）

```bash
# 将在Week 3准备
# 支持的云平台：
# - AWS (推荐): p3.2xlarge (V100 GPU)
# - Google Cloud: n1-standard-8 + T4 GPU
# - 阿里云: ecs.gn6i-c4g1.xlarge (T4 GPU)
```

---

### 1.2 创建RLlib多智能体环境（Day 2-4）

#### 文件结构

```
MARL-task/
├── rllib_env/
│   ├── __init__.py
│   ├── berth_allocation_env.py    # 核心环境
│   ├── utils.py                    # 工具函数
│   └── test_env.py                 # 测试脚本
├── environment/                     # 复用现有代码
│   ├── vessel.py
│   ├── shore_power.py
│   └── ...
└── rllib_train.py                  # 训练脚本
```

#### 核心代码：`rllib_env/berth_allocation_env.py`

```python
"""
RLlib多智能体泊位分配环境
"""
from typing import Dict, Optional
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# 导入现有组件
import sys
sys.path.append('..')
from environment.vessel import Vessel, VesselGenerator
from environment.shore_power import ShorePowerManager
from rewards.reward_calculator import RewardCalculator


class BerthAllocationMultiAgentEnv(MultiAgentEnv):
    """
    泊位分配多智能体环境 - RLlib适配

    特性:
    - 每艘船是一个智能体
    - 连续动作空间 (3维)
    - 部分可观测 (POMDP)
    - 个体奖励 (每艘船独立奖励)
    """

    def __init__(self, config: dict):
        super().__init__()

        # 环境参数
        self.berth_length = config.get('berth_length', 2000)
        self.max_vessels = config.get('max_vessels', 20)
        self.planning_horizon = config.get('planning_horizon', 168)  # 7天

        # 船舶生成器
        self.vessel_generator = VesselGenerator(config)

        # 岸电管理器
        if config.get('shore_power_enabled', True):
            self.shore_power_manager = ShorePowerManager(config)
        else:
            self.shore_power_manager = None

        # 奖励计算器
        self.reward_calculator = RewardCalculator(config)

        # 环境状态
        self.vessels = []
        self.allocations = []
        self.current_step = 0
        self.current_time = 0

        # 定义观测空间 (17维局部观测)
        self._obs_space = Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )

        # 定义动作空间 (3维连续动作)
        self._action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # RLlib要求的智能体ID集合
        self._agent_ids = set()

    @property
    def observation_space(self):
        """返回观测空间"""
        return self._obs_space

    @property
    def action_space(self):
        """返回动作空间"""
        return self._action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境

        Returns:
            observations: Dict[agent_id, obs]
            infos: Dict[agent_id, info]
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)

        # 生成船舶
        if options and 'vessels' in options:
            self.vessels = options['vessels']
        else:
            self.vessels = self.vessel_generator.generate_vessels(
                num_vessels=self.max_vessels
            )

        # 重置状态
        self.allocations = []
        self.current_step = 0
        self.current_time = 0

        # 设置智能体ID（每艘船一个ID）
        self._agent_ids = {f"vessel_{i}" for i in range(len(self.vessels))}

        # 获取初始观测
        observations = self._get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]):
        """
        执行动作

        Args:
            action_dict: Dict[agent_id, action]
                action: [position, wait_time, shore_power_prob]
                       范围: [-1, 1]

        Returns:
            observations: Dict[agent_id, obs]
            rewards: Dict[agent_id, float]
            terminateds: Dict[agent_id, bool]
            truncateds: Dict[agent_id, bool]
            infos: Dict[agent_id, dict]
        """
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        # 处理每个智能体的动作
        for agent_id, action in action_dict.items():
            vessel_idx = int(agent_id.split('_')[1])
            vessel = self.vessels[vessel_idx]

            # 解码动作 (从[-1,1]映射到实际范围)
            position = self._decode_position(action[0])
            wait_time = self._decode_wait_time(action[1])
            shore_power_prob = self._decode_shore_power(action[2])

            # 决定是否使用岸电
            uses_shore_power = (
                vessel.can_use_shore_power and
                np.random.random() < shore_power_prob
            )

            # 创建分配
            allocation = {
                'vessel': vessel,
                'vessel_id': vessel.id,
                'position': position,
                'berthing_time': vessel.arrival_time + wait_time,
                'departure_time': vessel.arrival_time + wait_time + vessel.operation_time,
                'uses_shore_power': uses_shore_power,
                'waiting_time': wait_time,
            }

            # 添加到分配列表
            self.allocations.append(allocation)

            # 计算奖励
            reward = self.reward_calculator.calculate_reward(
                allocation,
                self.allocations,
                self.shore_power_manager
            )

            # 获取观测
            obs = self._get_observation(vessel_idx)

            # 填充返回值
            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminateds[agent_id] = False  # 分批决策，不提前终止
            truncateds[agent_id] = False
            infos[agent_id] = {
                'allocated': True,
                'position': position,
                'berthing_time': allocation['berthing_time'],
                'uses_shore_power': uses_shore_power
            }

        # 更新时间步
        self.current_step += 1

        # 检查是否所有船舶都已分配
        if len(self.allocations) >= len(self.vessels):
            # Episode结束
            terminateds['__all__'] = True
            truncateds['__all__'] = False
        else:
            terminateds['__all__'] = False
            truncateds['__all__'] = False

        return observations, rewards, terminateds, truncateds, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有智能体的观测"""
        observations = {}
        for i, vessel in enumerate(self.vessels):
            agent_id = f"vessel_{i}"
            observations[agent_id] = self._get_observation(i)
        return observations

    def _get_observation(self, vessel_idx: int) -> np.ndarray:
        """
        获取单个智能体的观测 (17维)

        观测内容:
        - 静态特征 (4): 船长, 到港时间, 优先级, 岸电能力
        - 动态特征 (3): 当前时间, 等待时间, 操作时间
        - 岸电信息 (6): 5段使用率 + 总使用率
        - 泊位信息 (4): 左右邻近距离, 可用空间, 占用率
        """
        vessel = self.vessels[vessel_idx]

        # 静态特征（归一化）
        vessel_length = vessel.length / self.berth_length
        arrival_time = vessel.arrival_time / self.planning_horizon
        priority = vessel.priority / 4.0
        shore_power_cap = 1.0 if vessel.can_use_shore_power else 0.0

        # 动态特征
        current_time = self.current_time / self.planning_horizon
        waiting_time = max(0, self.current_time - vessel.arrival_time) / 48.0
        operation_time = vessel.operation_time / self.planning_horizon

        # 岸电使用率（如果有）
        if self.shore_power_manager:
            shore_power_usage = self.shore_power_manager.get_usage_rates()
        else:
            shore_power_usage = [0.0] * 6

        # 泊位信息（简化版）
        left_distance = 1.0  # 默认值
        right_distance = 1.0
        available_space = 1.0
        berth_occupancy = len(self.allocations) / self.max_vessels

        # 组合观测
        obs = np.array([
            vessel_length,
            arrival_time,
            priority,
            shore_power_cap,
            current_time,
            waiting_time,
            operation_time,
            *shore_power_usage,
            left_distance,
            right_distance,
            available_space,
            berth_occupancy,
        ], dtype=np.float32)

        return obs

    def _decode_position(self, action_value: float) -> float:
        """解码位置动作"""
        # action_value ∈ [-1, 1]
        # position ∈ [0, berth_length]
        position = (action_value + 1) * self.berth_length / 2
        return np.clip(position, 0, self.berth_length)

    def _decode_wait_time(self, action_value: float) -> float:
        """解码等待时间动作"""
        # action_value ∈ [-1, 1]
        # wait_time ∈ [0, 48]小时
        wait_time = (action_value + 1) * 24  # 0-48小时
        return np.clip(wait_time, 0, 48)

    def _decode_shore_power(self, action_value: float) -> float:
        """解码岸电概率"""
        # action_value ∈ [-1, 1]
        # probability ∈ [0, 1]
        prob = (action_value + 1) / 2
        return np.clip(prob, 0, 1)
```

---

### 1.3 测试环境（Day 4）

#### 测试脚本：`rllib_env/test_env.py`

```python
"""测试RLlib环境"""
import numpy as np
from berth_allocation_env import BerthAllocationMultiAgentEnv

# 配置
config = {
    'berth_length': 2000,
    'max_vessels': 5,  # 测试用小数量
    'planning_horizon': 168,
    'shore_power_enabled': True,
    'generation_mode': 'simple',
}

# 创建环境
env = BerthAllocationMultiAgentEnv(config)

# 重置
obs, info = env.reset()
print(f"初始观测: {len(obs)} 个智能体")
print(f"观测空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 运行几步
for step in range(5):
    # 随机动作
    actions = {
        agent_id: env.action_space.sample()
        for agent_id in obs.keys()
    }

    obs, rewards, dones, truncs, infos = env.step(actions)

    print(f"\n步骤 {step+1}:")
    print(f"  奖励: {rewards}")
    print(f"  完成: {dones}")

    if dones.get('__all__', False):
        print("Episode 完成!")
        break

print("\n✅ 环境测试通过!")
```

运行测试:

```bash
cd rllib_env
python test_env.py
```

---

## 🏗️ 阶段2: 多智能体算法配置（Week 2）

### 2.1 配置MADDPG算法（Day 1-3）

#### 训练脚本：`rllib_train.py`

```python
"""
RLlib训练脚本 - MADDPG算法
"""
import ray
from ray import tune
from ray.rllib.algorithms.maddpg import MADDPGConfig
from ray.rllib.policy.policy import PolicySpec
import os

# 导入环境
from rllib_env.berth_allocation_env import BerthAllocationMultiAgentEnv

# 初始化Ray
ray.init(num_gpus=1)  # 云端训练时使用GPU

# 环境配置
env_config = {
    'berth_length': 2000,
    'max_vessels': 20,
    'planning_horizon': 168,
    'shore_power_enabled': True,
    'generation_mode': 'realistic',
    'reward_weights': {
        'c1_base': 100.0,
        'c2_waiting': 10.0,
        'c3_emission': 0.01,
        'c4_shore_power': 50.0,
        'c5_utilization': 200.0,
        'c6_spacing': 30.0,
    }
}

# 创建临时环境获取智能体数量
temp_env = BerthAllocationMultiAgentEnv(env_config)
temp_env.reset()
num_agents = len(temp_env._agent_ids)
temp_env.close()

# 定义策略（每个智能体可以有独立策略）
policies = {
    f"policy_vessel_{i}": PolicySpec(
        policy_class=None,  # 使用默认策略
        observation_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        config={},
    )
    for i in range(num_agents)
}

# 策略映射函数
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """将智能体ID映射到策略"""
    # 简单策略：每个vessel使用对应的policy
    vessel_idx = int(agent_id.split('_')[1])
    return f"policy_vessel_{vessel_idx}"

# 配置MADDPG
config = (
    MADDPGConfig()
    .environment(
        env=BerthAllocationMultiAgentEnv,
        env_config=env_config,
    )
    .framework("torch")
    .training(
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.005,
        gamma=0.99,
        n_step=1,
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 1000000,
        },
        train_batch_size=256,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    .resources(
        num_gpus=1,          # 使用1个GPU
        num_cpus_per_worker=1,
    )
    .rollouts(
        num_rollout_workers=4,  # 4个并行采样器
        num_envs_per_worker=1,
    )
    .reporting(
        min_train_timesteps_per_iteration=1000,
        min_sample_timesteps_per_iteration=1000,
    )
)

# 训练
results = tune.run(
    "MADDPG",
    config=config.to_dict(),
    stop={
        "timesteps_total": 1000000,  # 1M steps
    },
    checkpoint_freq=10,
    checkpoint_at_end=True,
    local_dir="./ray_results",
    verbose=1,
)

print("训练完成!")
print(f"最佳检查点: {results.best_checkpoint}")
```

---

### 2.2 配置其他算法（Day 4-5）

#### MAPPO配置

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(env=BerthAllocationMultiAgentEnv, env_config=env_config)
    .framework("torch")
    .training(
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    .resources(num_gpus=1, num_cpus_per_worker=1)
    .rollouts(num_rollout_workers=4, num_envs_per_worker=1)
)
```

#### QMIX配置（如需要）

```python
from ray.rllib.algorithms.qmix import QMixConfig

# 注意：QMIX需要离散动作，需要修改环境
# 或使用离散化版本的泊位分配环境
```

---

## 🏗️ 阶段3: 云平台部署（Week 3）

### 3.1 准备Docker镜像（推荐）

#### Dockerfile

```dockerfile
FROM rayproject/ray:latest-gpu

# 安装依赖
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

# 复制代码
COPY . /workspace/
WORKDIR /workspace

# 启动命令
CMD ["python", "rllib_train.py"]
```

### 3.2 云平台选择

#### 推荐方案1: AWS EC2

```bash
# 实例类型: p3.2xlarge (V100 GPU, 8 vCPUs, 61GB RAM)
# 价格: ~$3/小时

# 启动实例
aws ec2 run-instances \
  --image-id ami-xxxxx \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx

# SSH连接
ssh -i your-key.pem ubuntu@ec2-xxx.compute.amazonaws.com

# 安装环境
git clone your-repo
cd MARL-task
pip install -r requirements.txt

# 运行训练
python rllib_train.py
```

#### 推荐方案2: Google Colab Pro（最简单）

```python
# Colab笔记本
!git clone https://github.com/your-username/MARL-task.git
%cd MARL-task

!pip install ray[rllib] torch

# 运行训练
!python rllib_train.py
```

#### 推荐方案3: 阿里云ECS

```bash
# 实例: ecs.gn6i-c4g1.xlarge (T4 GPU)
# 价格: ~¥8/小时

# 镜像: Ubuntu 20.04 + CUDA 11.8
```

---

## 🏗️ 阶段4: 大规模实验（Week 4-6）

### 4.1 实验设计

#### 实验矩阵

```python
# experiments/experiment_configs.py

SCENARIOS = {
    'small': {'max_vessels': 10, 'planning_horizon': 168},
    'medium': {'max_vessels': 15, 'planning_horizon': 168},
    'large': {'max_vessels': 20, 'planning_horizon': 168},
    'xlarge': {'max_vessels': 25, 'planning_horizon': 168},
}

ALGORITHMS = ['MADDPG', 'MAPPO', 'IPPO']

CONFIGURATIONS = {
    'shore_power': [True, False],
    'generation_mode': ['realistic', 'simple'],
}

SEEDS = [42, 123, 456, 789, 1024]

# 总实验数
total_experiments = (
    len(SCENARIOS) *
    len(ALGORITHMS) *
    len(CONFIGURATIONS['shore_power']) *
    len(CONFIGURATIONS['generation_mode']) *
    len(SEEDS)
)
# = 4 * 3 * 2 * 2 * 5 = 240组实验
```

### 4.2 批量训练脚本

```python
# experiments/run_all_experiments.py

import itertools
import subprocess
import json

# 生成所有实验组合
experiments = []
for scenario, algo, shore_power, gen_mode, seed in itertools.product(
    SCENARIOS.keys(),
    ALGORITHMS,
    CONFIGURATIONS['shore_power'],
    CONFIGURATIONS['generation_mode'],
    SEEDS
):
    exp_config = {
        'scenario': scenario,
        'algorithm': algo,
        'shore_power': shore_power,
        'generation_mode': gen_mode,
        'seed': seed,
        'name': f"{algo}_{scenario}_sp{shore_power}_gm{gen_mode}_s{seed}"
    }
    experiments.append(exp_config)

print(f"总共 {len(experiments)} 个实验")

# 运行所有实验
for i, exp in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] 运行: {exp['name']}")

    # 构建命令
    cmd = [
        "python", "rllib_train.py",
        "--algo", exp['algorithm'],
        "--scenario", exp['scenario'],
        "--shore-power", str(exp['shore_power']),
        "--gen-mode", exp['generation_mode'],
        "--seed", str(exp['seed']),
        "--name", exp['name'],
    ]

    # 运行
    subprocess.run(cmd)

print("\n✅ 所有实验完成!")
```

### 4.3 预计训练时间

```python
# 单个实验（1M steps）:
# - CPU: 4-6小时
# - GPU (T4): 30-60分钟
# - GPU (V100): 15-30分钟

# 240个实验:
# - 串行 (V100): 240 * 0.5h = 120小时 = 5天
# - 4个GPU并行: 120h / 4 = 30小时 ≈ 1.5天

# 预算估算（AWS p3.2xlarge, $3/小时）:
# 240实验 * 0.5h * $3 = $360
# 或使用Spot实例: ~$100-150
```

---

## 📊 阶段5: 结果分析（Week 6）

### 5.1 提取训练数据

```python
# analysis/extract_results.py

from ray.tune import Analysis
import pandas as pd

# 加载所有实验结果
analysis = Analysis("./ray_results")

# 提取关键指标
results = []
for trial in analysis.trials:
    config = trial.config
    last_result = trial.last_result

    results.append({
        'algorithm': config['algorithm'],
        'scenario': config['scenario'],
        'shore_power': config['shore_power'],
        'seed': config['seed'],
        'reward_mean': last_result.get('episode_reward_mean'),
        'berth_utilization': last_result.get('custom_metrics/berth_utilization'),
        'avg_waiting_time': last_result.get('custom_metrics/avg_waiting_time'),
        'total_emissions': last_result.get('custom_metrics/total_emissions'),
    })

df = pd.DataFrame(results)
df.to_csv('results_summary.csv', index=False)
print(df.groupby(['algorithm', 'scenario']).mean())
```

### 5.2 绘图

```python
# analysis/plot_results.py

import matplotlib.pyplot as plt
import seaborn as sns

# 1. 训练曲线对比
for algo in ALGORITHMS:
    algo_data = df[df['algorithm'] == algo]
    plt.plot(algo_data['timesteps'], algo_data['reward'], label=algo)
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.savefig('training_curves.pdf')

# 2. 性能对比柱状图
sns.barplot(data=df, x='algorithm', y='berth_utilization', hue='scenario')
plt.savefig('performance_comparison.pdf')

# 3. 雷达图（多指标）
# ...

print("图表已生成!")
```

---

## 📝 阶段6: 论文撰写（Week 7-12）

（与之前方案类似，但强调RLlib的优势）

### 论文结构

```
I. Introduction
   - 强调：使用工业级MARL框架（RLlib）

II. Related Work
   - 包括：Ray RLlib在MARL中的应用

III. Problem Formulation
   (与之前相同)

IV. Methodology
   A. MADDPG算法
   B. RLlib实现细节
   C. 分布式训练策略

V. Experiments
   A. 实验环境（RLlib + GPU集群）
   B. 基准对比（MADDPG vs MAPPO vs IPPO）
   C. 消融实验

VI. Results
   - 强调：大规模实验（240组）
   - 强调：训练效率（GPU加速）

VII. Conclusion
```

---

## 🎯 关键优势总结

### RLlib带来的优势

1. **训练速度** ⚡
   - GPU加速: 比CPU快10-20倍
   - 分布式: 4个GPU并行，总时间缩短75%

2. **实验规模** 📊
   - 240组大规模实验可行
   - 完整的消融研究

3. **代码质量** 💻
   - 工业级框架，少Bug
   - 详细文档，易调试

4. **未来价值** 🚀
   - 训练好的模型可直接部署
   - 论文+实际应用双重价值

---

## ✅ 立即行动计划

### 本周任务（Week 1）

**Day 1**:
- [ ] 安装Ray和RLlib
- [ ] 验证环境

**Day 2-3**:
- [ ] 实现`BerthAllocationMultiAgentEnv`
- [ ] 复用现有environment代码

**Day 4**:
- [ ] 测试环境
- [ ] 修复Bug

**Day 5**:
- [ ] 创建MADDPG训练脚本
- [ ] 本地小规模测试

**周末**:
- [ ] 准备云平台账号
- [ ] 预算规划

---

## 📋 需要确认

1. **云平台选择**:
   - [ ] AWS（推荐，灵活）
   - [ ] Google Colab Pro（最简单，$10/月）
   - [ ] 阿里云（国内快）

2. **预算**:
   - 实验成本: $100-400
   - 能否接受？

3. **立即开始**？
   - 我可以现在就帮您创建第一个RLlib环境文件！

**请告诉我您的决定，我们立即开始实施！** 🚀
