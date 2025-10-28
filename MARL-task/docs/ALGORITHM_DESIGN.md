# 算法设计完整文档

**多智能体强化学习泊位分配与岸电协同优化**

---

## 📋 目录

- [第一部分：问题建模](#第一部分问题建模)
- [第二部分：环境设计](#第二部分环境设计)
- [第三部分：奖励函数](#第三部分奖励函数)
- [第四部分：算法实现](#第四部分算法实现)
- [第五部分：改进与优化](#第五部分改进与优化)

---

## 第一部分：问题建模

### 1.1 问题描述

**核心目标**: 为到港船舶智能分配泊位位置、靠泊时间和岸电使用，实现多目标优化

**优化目标**:
1. 最小化船舶等待时间
2. 最大化泊位利用率
3. 降低碳排放 (鼓励使用岸电)
4. 避免船舶拥挤
5. 满足运营约束

### 1.2 MARL建模

**建模框架**: 部分可观测马尔可夫决策过程 (POMDP)

**关键要素**:
- **智能体**: 每艘船是一个独立智能体
- **动作**: 连续动作 [泊位位置, 等待时间, 岸电概率]
- **观测**: 局部观测 (17维特征)
- **奖励**: 多目标奖励函数
- **终止**: 所有船舶完成靠泊

**CTDE架构** (Centralized Training, Decentralized Execution):
- 训练时: 使用全局信息
- 执行时: 基于局部观测决策

---

## 第二部分：环境设计

### 2.1 观测空间 (17维)

```python
observation_space = Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
```

**维度说明**:

| 维度 | 特征 | 描述 | 归一化方法 |
|------|------|------|-----------|
| 0 | 泊位位置 | 当前分配的位置 | [0, berth_length] → [-1, 1] |
| 1 | 等待时间 | 已等待时间 | [0, max_waiting] → [-1, 1] |
| 2 | 岸电概率 | 岸电使用概率 | [0, 1] → [-1, 1] |
| 3 | 船舶长度 | 船长 | [min_len, max_len] → [-1, 1] |
| 4 | 到港时间 | 到达时间 | [0, horizon] → [-1, 1] |
| 5 | 优先级 | 船舶优先级 | [0, 1] → [-1, 1] |
| 6-10 | 岸电负载 | 5段岸电使用率 | [0, capacity] → [-1, 1] |
| 11 | 泊位利用率 | 当前利用率 | [0, 1] → [-1, 1] |
| 12 | 平均等待 | 平均等待时间 | [0, max_waiting] → [-1, 1] |
| 13 | 碳排放 | 累计排放量 | [0, max_emission] → [-1, 1] |
| 14 | 岸电使用率 | 岸电使用比例 | [0, 1] → [-1, 1] |
| 15 | 成功率 | 分配成功率 | [0, 1] → [-1, 1] |
| 16 | 时间步 | 当前时间步 | [0, max_steps] → [-1, 1] |

**观测获取**:
```python
def _get_obs(self, vessel_id):
    """获取智能体观测"""
    vessel = self.vessels[vessel_id]

    obs = np.array([
        # 当前动作状态 (0-2)
        normalize(vessel.position, 0, self.berth_length),
        normalize(vessel.waiting_time, 0, self.max_waiting),
        vessel.shore_power_prob,

        # 船舶特征 (3-5)
        normalize(vessel.length, self.min_vessel_len, self.max_vessel_len),
        normalize(vessel.arrival_time, 0, self.planning_horizon),
        vessel.priority,

        # 岸电状态 (6-10)
        *[normalize(load, 0, capacity) for load, capacity in
          zip(self.shore_power.loads, self.shore_power.capacities)],

        # 全局指标 (11-15)
        self.berth_utilization,
        normalize(self.avg_waiting_time, 0, self.max_waiting),
        normalize(self.total_emissions, 0, self.max_emissions),
        self.shore_power_usage_rate,
        self.allocation_success_rate,

        # 时间进度 (16)
        normalize(self.current_step, 0, self.max_steps),
    ], dtype=np.float32)

    return np.clip(obs, -1.0, 1.0)
```

### 2.2 动作空间 (3维连续)

```python
action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

**动作维度**:

| 维度 | 动作 | 原始范围 | 映射方法 |
|------|------|---------|----------|
| 0 | 泊位位置 | [0, berth_length] | `(a[0]+1)/2 * berth_length` |
| 1 | 等待时间 | [0, max_waiting] | `(a[1]+1)/2 * max_waiting` |
| 2 | 岸电概率 | [0, 1] | `(a[2]+1)/2` |

**动作映射**:
```python
def map_action(self, raw_action):
    """将[-1,1]动作映射到实际范围"""
    position = (raw_action[0] + 1) / 2 * self.berth_length
    waiting_time = (raw_action[1] + 1) / 2 * self.max_waiting_time
    shore_power_prob = (raw_action[2] + 1) / 2

    return {
        'position': np.clip(position, 0, self.berth_length),
        'waiting_time': np.clip(waiting_time, 0, self.max_waiting_time),
        'shore_power_prob': np.clip(shore_power_prob, 0, 1),
    }
```

**动作约束**:
```python
def check_action_valid(self, vessel, action):
    """检查动作是否有效"""
    position, waiting_time, _ = action

    # 约束1: 位置在泊位范围内
    if position < 0 or position + vessel.length > self.berth_length:
        return False

    # 约束2: 等待时间不超过限制
    if waiting_time > self.max_waiting_time:
        return False

    # 约束3: 不与其他船冲突
    for other_vessel in self.allocated_vessels:
        if self._check_collision(vessel, other_vessel, position):
            return False

    return True
```

### 2.3 船舶生成

**非齐次泊松过程**:
```python
class VesselGenerator:
    def generate_realistic(self, num_vessels, horizon_days):
        """生成真实船舶序列"""
        vessels = []

        # 非齐次泊松过程 - 多峰到港
        for t in range(horizon_days * 24):  # 按小时
            # 到港率随时间变化
            rate = self._get_arrival_rate(t)

            # 生成泊松数量
            num_arrivals = np.random.poisson(rate)

            for _ in range(num_arrivals):
                vessel = self._create_vessel(t)
                vessels.append(vessel)

        return vessels[:num_vessels]

    def _get_arrival_rate(self, hour):
        """到港率函数 (多峰)"""
        # 早高峰 (6-10时)
        peak1 = 2.0 if 6 <= hour % 24 <= 10 else 0.5

        # 晚高峰 (18-22时)
        peak2 = 1.5 if 18 <= hour % 24 <= 22 else 0.5

        return peak1 + peak2

    def _create_vessel(self, arrival_time):
        """创建船舶"""
        # 多峰船长分布
        vessel_type = np.random.choice(['small', 'medium', 'large'],
                                      p=[0.3, 0.5, 0.2])

        if vessel_type == 'small':
            length = np.random.normal(50, 10)
        elif vessel_type == 'medium':
            length = np.random.normal(150, 20)
        else:  # large
            length = np.random.normal(300, 30)

        return Vessel(
            length=np.clip(length, 30, 400),
            arrival_time=arrival_time,
            priority=np.random.uniform(0, 1),
            shore_power_capable=np.random.rand() > 0.3,
        )
```

### 2.4 岸电管理

```python
class ShorePowerManager:
    def __init__(self, num_segments=5):
        """5段岸电管理"""
        self.num_segments = num_segments
        self.segment_length = 2000 / num_segments  # 400m/段

        # 每段容量 (kW)
        self.capacities = [500, 500, 500, 500, 500]
        self.loads = [0] * num_segments

    def allocate(self, vessel, position, shore_power_prob):
        """分配岸电"""
        # 确定船舶占用哪些段
        start_seg = int(position / self.segment_length)
        end_seg = int((position + vessel.length) / self.segment_length)

        # 检查容量
        for seg in range(start_seg, end_seg + 1):
            if seg < self.num_segments:
                required = vessel.power_demand * shore_power_prob
                if self.loads[seg] + required > self.capacities[seg]:
                    return False, 0

        # 分配成功
        allocated_power = 0
        for seg in range(start_seg, end_seg + 1):
            if seg < self.num_segments:
                power = vessel.power_demand * shore_power_prob
                self.loads[seg] += power
                allocated_power += power

        return True, allocated_power

    def calculate_emissions(self, vessel, shore_power_used):
        """计算碳排放"""
        # 使用岸电部分: 较低排放
        shore_power_emission = shore_power_used * 0.5  # kg CO2/kWh

        # 未使用岸电: 船舶自发电
        remaining_power = vessel.power_demand - shore_power_used
        vessel_emission = remaining_power * 0.8  # kg CO2/kWh

        total = shore_power_emission + vessel_emission
        reduction = vessel.power_demand * 0.8 - total  # 减排量

        return total, reduction
```

---

## 第三部分：奖励函数

### 3.1 多目标奖励设计

```python
reward = c1 * base_reward          # 成功靠泊基础奖励
       - c2 * waiting_penalty      # 等待时间惩罚
       - c3 * emission_penalty     # 碳排放惩罚
       + c4 * shore_power_bonus    # 岸电使用奖励
       + c5 * utilization_reward   # 泊位利用率奖励
       + c6 * spacing_reward       # 分散靠泊奖励
```

**默认权重**:
```python
reward_weights = {
    'c1': 1.0,   # 基础奖励
    'c2': 0.5,   # 等待惩罚
    'c3': 0.3,   # 排放惩罚
    'c4': 0.4,   # 岸电奖励
    'c5': 0.6,   # 利用率奖励
    'c6': 0.2,   # 分散奖励
}
```

### 3.2 奖励分项详解

#### 3.2.1 基础奖励
```python
def base_reward(self, vessel):
    """成功靠泊的基础奖励"""
    if self.is_allocated_successfully(vessel):
        # 根据优先级加权
        reward = 10.0 * vessel.priority
        return reward
    else:
        return 0.0
```

#### 3.2.2 等待时间惩罚
```python
def waiting_penalty(self, vessel):
    """等待时间惩罚 (凸函数)"""
    waiting_hours = vessel.waiting_time

    # 分段惩罚
    if waiting_hours <= 2:
        penalty = waiting_hours * 0.5
    elif waiting_hours <= 6:
        penalty = 1.0 + (waiting_hours - 2) * 1.0
    else:
        penalty = 5.0 + (waiting_hours - 6) * 2.0

    return penalty
```

#### 3.2.3 碳排放惩罚
```python
def emission_penalty(self, vessel, shore_power_used):
    """碳排放惩罚"""
    total_emission, reduction = self.shore_power.calculate_emissions(
        vessel, shore_power_used
    )

    # 归一化排放量
    normalized_emission = total_emission / vessel.max_possible_emission

    penalty = normalized_emission * 5.0
    return penalty
```

#### 3.2.4 岸电使用奖励
```python
def shore_power_bonus(self, vessel, shore_power_used):
    """岸电使用奖励"""
    # 使用比例
    usage_ratio = shore_power_used / vessel.power_demand

    # 非线性奖励 (鼓励高使用率)
    bonus = usage_ratio ** 2 * 3.0

    return bonus
```

#### 3.2.5 泊位利用率奖励
```python
def utilization_reward(self):
    """泊位利用率奖励"""
    # 时空利用率
    utilization = self.calculate_berth_utilization()

    # 目标利用率: 80-90%
    if 0.8 <= utilization <= 0.9:
        reward = 5.0
    elif utilization > 0.9:
        # 过高利用率轻微惩罚 (可能拥挤)
        reward = 5.0 - (utilization - 0.9) * 10.0
    else:
        # 低利用率惩罚
        reward = utilization * 6.25  # 0.8*6.25=5.0

    return reward
```

#### 3.2.6 分散靠泊奖励 (改进版)
```python
def spacing_reward(self, vessel, position):
    """分散靠泊奖励 - 基于邻近拥挤度"""
    # 原版: 基于岸线中心距离 (不合理)
    # 改进: 基于邻近船舶拥挤度

    # 计算邻近拥挤度
    neighbor_density = 0
    search_radius = 200  # 200m范围内

    for other_vessel in self.allocated_vessels:
        distance = abs(other_vessel.position - position)

        if distance < search_radius:
            # 距离越近,拥挤度越高
            density_contribution = (search_radius - distance) / search_radius
            neighbor_density += density_contribution

    # 拥挤度越低,奖励越高
    reward = max(0, 2.0 - neighbor_density)

    return reward
```

### 3.3 完整奖励计算

```python
class RewardCalculator:
    def __init__(self, weights):
        self.weights = weights

    def calculate_reward(self, vessel, action, state):
        """计算完整奖励"""
        position, waiting_time, shore_power_prob = action

        # 检查动作有效性
        if not self.is_valid_action(vessel, action):
            return -10.0  # 无效动作大惩罚

        # 分项计算
        r1 = self.base_reward(vessel)
        r2 = self.waiting_penalty(vessel)
        r3 = self.emission_penalty(vessel, shore_power_prob)
        r4 = self.shore_power_bonus(vessel, shore_power_prob)
        r5 = self.utilization_reward()
        r6 = self.spacing_reward(vessel, position)

        # 加权求和
        total_reward = (
            self.weights['c1'] * r1
            - self.weights['c2'] * r2
            - self.weights['c3'] * r3
            + self.weights['c4'] * r4
            + self.weights['c5'] * r5
            + self.weights['c6'] * r6
        )

        return total_reward

    def get_reward_breakdown(self):
        """返回奖励分解 (用于分析)"""
        return {
            'base_reward': self.last_r1,
            'waiting_penalty': self.last_r2,
            'emission_penalty': self.last_r3,
            'shore_power_bonus': self.last_r4,
            'utilization_reward': self.last_r5,
            'spacing_reward': self.last_r6,
            'total': self.last_total,
        }
```

---

## 第四部分：算法实现

### 4.1 SAC算法 (推荐)

**算法特点**:
- 最大熵强化学习
- Off-policy
- Twin Q-networks
- 自动温度调节
- 适合连续动作空间

**核心公式**:

**策略目标**:
```
J(π) = E[∑ γ^t (r_t + α H(π(·|s_t)))]

H(π) = -E[log π(a|s)]  # 熵
```

**Q函数更新**:
```
Q(s,a) ← r + γ min(Q₁(s',a'), Q₂(s',a'))  # Twin Q
```

**策略更新**:
```
π ← arg max E[Q(s,a) - α log π(a|s)]
```

**RLlib配置**:
```python
from ray.rllib.algorithms.sac import SACConfig

config = SACConfig()

config.training(
    # Replay buffer
    train_batch_size=256,
    replay_buffer_config={
        "type": "MultiAgentReplayBuffer",
        "capacity": 500000,
    },

    # Optimization
    optimization={
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },

    # SAC specific
    tau=0.005,                    # 软更新系数
    target_network_update_freq=1,
    twin_q=True,                  # 双Q网络
    policy_delay=2,               # 延迟策略更新

    # 探索
    initial_alpha=1.0,
    target_entropy="auto",        # 自动调节温度
)

# 网络架构
config.model = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
}
```

### 4.2 PPO算法

**算法特点**:
- On-policy
- 裁剪目标函数
- 广义优势估计(GAE)
- 稳定可靠

**核心公式**:

**裁剪目标**:
```
L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

r(θ) = π_θ(a|s) / π_old(a|s)  # 重要性权重
```

**优势函数 (GAE)**:
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**RLlib配置**:
```python
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()

config.training(
    # Batch配置
    train_batch_size=4000,
    sgd_minibatch_size=128,
    num_sgd_iter=10,

    # PPO参数
    lr=3e-4,
    gamma=0.99,
    lambda_=0.95,
    clip_param=0.2,
    vf_clip_param=10.0,
    entropy_coeff=0.01,

    # GAE
    use_gae=True,
    use_critic=True,
)
```

### 4.3 算法对比

| 特性 | SAC | PPO | 推荐场景 |
|------|-----|-----|---------|
| **样本效率** | 高 | 中 | SAC适合样本昂贵 |
| **训练稳定性** | 中 | 高 | PPO适合baseline |
| **超参数敏感** | 低 | 中 | SAC更鲁棒 |
| **连续动作** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SAC最优 |
| **计算开销** | 中 | 低 | PPO更快 |

---

## 第五部分:改进与优化

### 5.1 论文评审改进

**改进1: 移除状态噪声**
```python
# 原版 (错误): 17维观测 + 1维噪声 = 18维
obs = [..., noise]

# 改进: 仅17维观测,噪声仅在动作层
obs = [...]  # 17维
action = policy(obs) + exploration_noise  # 噪声在这里
```

**改进2: 分散靠泊奖励**
```python
# 原版: 基于到岸线中心的距离
distance_to_center = abs(position - berth_length/2)
reward = -distance_to_center

# 改进: 基于邻近拥挤度
neighbor_density = sum(1/(1+dist) for dist in neighbor_distances)
reward = max(0, 2.0 - neighbor_density)
```

**改进3: 船舶生成现实化**
```python
# 原版: 均匀分布
vessels = [create_vessel(t) for t in uniform(0, horizon)]

# 改进: 非齐次泊松 + 多峰分布
vessels = generate_realistic_vessels(horizon)
```

### 5.2 奖励权重敏感性分析

```python
# 权重扫描实验
weight_configs = [
    {'c1': 1.0, 'c2': 0.3, 'c3': 0.3, 'c4': 0.4, 'c5': 0.6, 'c6': 0.2},
    {'c1': 1.0, 'c2': 0.5, 'c3': 0.3, 'c4': 0.4, 'c5': 0.6, 'c6': 0.2},
    {'c1': 1.0, 'c2': 0.7, 'c3': 0.3, 'c4': 0.4, 'c5': 0.6, 'c6': 0.2},
    # ... 更多配置
]

results = []
for weights in weight_configs:
    reward_calc = RewardCalculator(weights)
    performance = train_and_evaluate(reward_calc)
    results.append({'weights': weights, 'performance': performance})

# 分析最优权重
best_config = max(results, key=lambda x: x['performance'])
```

### 5.3 网络架构优化

```python
# 标准架构
config.model = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
}

# 深层架构 (更强表达能力)
config.model = {
    "fcnet_hiddens": [512, 512, 256],
    "fcnet_activation": "relu",
    "vf_share_layers": False,  # 独立价值网络
}

# 自定义架构
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomNetwork(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(17, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 策略头
        self.policy_head = nn.Linear(256, num_outputs)

        # 价值头
        self.value_head = nn.Linear(256, 1)
```

### 5.4 课程学习

```python
# 从简单到复杂的课程
curriculum = [
    {'num_vessels': 5, 'iterations': 500},
    {'num_vessels': 10, 'iterations': 500},
    {'num_vessels': 20, 'iterations': 500},
    {'num_vessels': 50, 'iterations': 1000},
    {'num_vessels': 100, 'iterations': 2000},
]

for stage in curriculum:
    env_config.update({'num_vessels': stage['num_vessels']})
    train(iterations=stage['iterations'])
```

---

## 附录A：评估指标

### A.1 性能指标

```python
def evaluate_performance(env, algo, num_episodes=100):
    """评估算法性能"""
    metrics = {
        'total_rewards': [],
        'waiting_times': [],
        'berth_utilization': [],
        'emissions': [],
        'shore_power_usage': [],
        'allocation_success_rate': [],
    }

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = {
                agent_id: algo.compute_single_action(obs[agent_id])
                for agent_id in obs.keys()
            }
            obs, rewards, dones, truncs, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            done = dones["__all__"]

        # 收集指标
        metrics['total_rewards'].append(episode_reward)
        metrics['waiting_times'].append(env.avg_waiting_time)
        metrics['berth_utilization'].append(env.berth_utilization)
        metrics['emissions'].append(env.total_emissions)
        metrics['shore_power_usage'].append(env.shore_power_usage_rate)
        metrics['allocation_success_rate'].append(env.success_rate)

    # 统计
    return {
        key: {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
        for key, values in metrics.items()
    }
```

### A.2 对比基线

```python
# Greedy基线
def greedy_baseline(env):
    """贪心算法: FCFS + 最早可用位置"""
    vessels = sorted(env.vessels, key=lambda v: v.arrival_time)

    for vessel in vessels:
        # 找最早可用位置
        position = find_earliest_available_position(vessel)
        env.allocate(vessel, position, waiting_time=0, shore_power_prob=1.0)

# FCFS基线
def fcfs_baseline(env):
    """先到先服务"""
    vessels = sorted(env.vessels, key=lambda v: v.arrival_time)

    for vessel in vessels:
        position = 0  # 总是从0开始
        env.allocate(vessel, position, waiting_time=0, shore_power_prob=0)
```

---

## 附录B：实验设置

### B.1 默认参数

```python
default_config = {
    # 环境
    'num_vessels': 50,
    'planning_horizon_days': 7,
    'berth_length': 2000.0,
    'max_waiting_time': 24.0,

    # 奖励权重
    'c1': 1.0,
    'c2': 0.5,
    'c3': 0.3,
    'c4': 0.4,
    'c5': 0.6,
    'c6': 0.2,

    # 训练
    'num_iterations': 1000,
    'batch_size': 256,
    'learning_rate': 3e-4,
}
```

### B.2 实验场景

| 场景 | 船舶数 | 目标 |
|------|--------|------|
| 小规模 | 10-20 | 算法验证 |
| 中规模 | 50 | 性能对比 |
| 大规模 | 100 | 可扩展性 |
| 超大规模 | 200+ | 极限测试 |

---

**文档维护**: Duan
**最后更新**: 2025-10-28
**版本**: v2.0
