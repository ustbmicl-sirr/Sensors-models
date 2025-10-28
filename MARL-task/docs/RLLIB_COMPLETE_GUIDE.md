# RLlib多智能体强化学习完整指南

**Ray RLlib 2.50.1 - 从入门到精通**

---

## 📋 目录

- [第一部分：RLlib框架原理](#第一部分rllib框架原理)
- [第二部分：环境实现](#第二部分环境实现)
- [第三部分：训练配置](#第三部分训练配置)
- [第四部分：并行优化](#第四部分并行优化)
- [第五部分：分布式训练](#第五部分分布式训练)
- [第六部分：实战案例](#第六部分实战案例)

---

## 第一部分：RLlib框架原理

### 1.1 RLlib架构概览

```
┌─────────────────────────────────────────────────────────┐
│                     RLlib完整架构                         │
└─────────────────────────────────────────────────────────┘

用户层: Python脚本 / CLI / Tune API
   ↓
算法层: PPO / SAC / DQN / A3C / APPO
   ↓
执行层: Rollout Workers / Training Iterator / Evaluation
   ↓
环境层: Gymnasium / MultiAgentEnv / Custom Env
   ↓
基础层: Ray Core (Actor / Object Store / Task Queue)
```

### 1.2 训练循环

```python
# RLlib训练循环伪代码
class Algorithm:
    def train(self):
        # 1. 并行采样
        samples = []
        for worker in self.workers:
            sample = worker.sample()  # 并行执行
            samples.append(sample)

        # 2. 聚合数据
        train_batch = concat(samples)

        # 3. GPU训练
        for minibatch in shuffle(train_batch):
            loss = self.compute_loss(minibatch)
            self.optimizer.step(loss)

        # 4. 更新策略
        self.update_target_networks()

        return metrics
```

### 1.3 Worker架构

```
Driver Process (主进程)
    │
    ├── Worker 0 (local)  → Env 0
    ├── Worker 1 (remote) → Env 1
    └── Worker N (remote) → Env N

每个Worker:
- 独立环境实例
- 独立策略副本 (inference模式)
- 收集样本数据
- 不进行梯度计算
```

---

## 第二部分：环境实现

### 2.1 MultiAgentEnv基类

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box
import numpy as np

class BerthAllocationMultiAgentEnv(MultiAgentEnv):
    """泊位分配多智能体环境"""

    def __init__(self, config):
        super().__init__()

        # 环境参数
        self.num_vessels = config.get("num_vessels", 10)
        self.berth_length = config.get("berth_length", 2000.0)

        # 观测空间: 17维连续
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # 动作空间: 3维连续
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._agent_ids = set()

    def reset(self, *, seed=None, options=None):
        """重置环境"""
        self._agent_ids = {f"vessel_{i}" for i in range(self.num_vessels)}

        observations = {
            agent_id: self._get_obs(agent_id)
            for agent_id in self._agent_ids
        }
        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, infos

    def step(self, action_dict):
        """执行一步"""
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for agent_id, action in action_dict.items():
            obs, reward, done, truncated, info = self._step_agent(
                agent_id, action
            )
            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminateds[agent_id] = done
            truncateds[agent_id] = truncated
            infos[agent_id] = info

        # 全局终止条件
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return observations, rewards, terminateds, truncateds, infos
```

### 2.2 环境注册

```python
from ray.tune.registry import register_env

def register_berth_env():
    def env_creator(env_config):
        return BerthAllocationMultiAgentEnv(env_config)

    register_env("berth_allocation", env_creator)
```

### 2.3 多智能体策略模式

**模式1: 共享策略** (参数共享,训练快)
```python
config.multi_agent(
    policies={
        "shared_policy": (None, obs_space, act_space, {})
    },
    policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
)
```

**模式2: 独立策略** (最大灵活性)
```python
config.multi_agent(
    policies={
        f"policy_{i}": (None, obs_space, act_space, {})
        for i in range(num_agents)
    },
    policy_mapping_fn=lambda agent_id, *args, **kwargs: f"policy_{agent_id}"
)
```

**模式3: 分组策略** (平衡)
```python
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    agent_idx = int(agent_id.split("_")[1])
    return "group_a" if agent_idx < 5 else "group_b"

config.multi_agent(
    policies={
        "group_a": (None, obs_space, act_space, {}),
        "group_b": (None, obs_space, act_space, {}),
    },
    policy_mapping_fn=policy_mapping_fn
)
```

---

## 第三部分：训练配置

### 3.1 基础配置模板

```python
from ray.rllib.algorithms.sac import SACConfig

# 创建配置
config = SACConfig()

# 环境配置
config.environment(
    env="berth_allocation",
    env_config={
        "num_vessels": 50,
        "planning_horizon_days": 7,
        "berth_length": 2000.0,
    }
)

# 框架配置
config.framework("torch")

# 资源配置
config.resources(
    num_gpus=1,
    num_cpus_for_driver=2,
)

# Rollout配置
config.rollouts(
    num_rollout_workers=8,
    num_envs_per_worker=2,
    rollout_fragment_length=500,
)

# 训练配置
config.training(
    train_batch_size=8000,
    replay_buffer_config={"capacity": 500000},
    optimization={
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
    },
)

# 评估配置
config.evaluation(
    evaluation_interval=10,
    evaluation_duration=10,
)

# 构建算法
algo = config.build()
```

### 3.2 SAC算法配置

```python
config = SACConfig()

config.training(
    train_batch_size=8000,
    replay_buffer_config={
        "type": "MultiAgentReplayBuffer",
        "capacity": 500000,
    },
    optimization={
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },
    tau=0.005,
    target_network_update_freq=1,
    twin_q=True,
    policy_delay=2,
)
```

### 3.3 PPO算法配置

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()

config.training(
    train_batch_size=8000,
    sgd_minibatch_size=256,
    num_sgd_iter=10,
    lr=3e-4,
    gamma=0.99,
    lambda_=0.95,
    clip_param=0.2,
    vf_clip_param=10.0,
    entropy_coeff=0.01,
)
```

### 3.4 完整训练脚本

```python
#!/usr/bin/env python3
import ray
from ray.rllib.algorithms.sac import SACConfig

# 初始化Ray
ray.init(num_gpus=1)

# 注册环境
register_berth_env()

# 配置
config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 50})
config.resources(num_gpus=1)
config.rollouts(num_rollout_workers=8)

# 训练
algo = config.build()
best_reward = -float('inf')

for i in range(1000):
    result = algo.train()

    reward = result['episode_reward_mean']
    print(f"Iteration {i}: Reward={reward:.2f}")

    if reward > best_reward:
        best_reward = reward
        checkpoint = algo.save()
        print(f"  Saved checkpoint: {checkpoint}")

algo.stop()
ray.shutdown()
```

---

## 第四部分：并行优化

### 4.1 数据并行 (核心机制)

```python
# 数据并行配置
config.rollouts(
    num_rollout_workers=16,      # 16个worker并行采样
    num_envs_per_worker=2,       # 每worker 2环境
    rollout_fragment_length=500, # 每次500步
)

# 数据流:
# 16 workers × 2 envs × 500 steps = 16000 样本/迭代
```

**工作流程**:
```
时间步 t:
Worker 1-16 并行采样 → 聚合16000样本 → GPU训练 → 更新策略
```

### 4.2 自动资源优化

```python
import multiprocessing
import torch

def get_optimal_config():
    """自动计算最优配置"""
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # 计算最优worker数
    if num_gpus > 0:
        num_workers = min(num_cpus - 2, num_gpus * 8)
    else:
        num_workers = max(2, num_cpus // 2)

    # 计算batch大小
    num_envs = 2
    fragment_length = 500 if num_gpus > 0 else 200
    train_batch_size = num_workers * num_envs * fragment_length

    return {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "train_batch_size": train_batch_size,
    }

# 使用
optimal = get_optimal_config()
config.resources(num_gpus=optimal["num_gpus"])
config.rollouts(num_rollout_workers=optimal["num_workers"])
config.training(train_batch_size=optimal["train_batch_size"])
```

### 4.3 GPU优化

```python
# 单GPU优化
config.resources(
    num_gpus=1,
    num_cpus_for_driver=2,
)
config.training(
    sgd_minibatch_size=512,  # 大batch提高GPU利用率
    num_sgd_iter=10,         # 多次迭代
)

# 多GPU learner
config.resources(num_gpus=0)  # Driver不用GPU
config.learners(
    num_learner_workers=4,
    num_gpus_per_learner_worker=1,
)
```

### 4.4 性能优化配置

```python
# 速度优先配置
config.rollouts(
    num_rollout_workers=32,
    num_envs_per_worker=4,
    rollout_fragment_length=2000,
    compress_observations=True,  # 压缩观测
)
config.training(
    train_batch_size=128000,
    replay_buffer_config={"capacity": 500000},
)

# 质量优先配置
config.rollouts(
    num_rollout_workers=8,
    num_envs_per_worker=1,
    rollout_fragment_length=500,
)
config.training(
    train_batch_size=4000,
    replay_buffer_config={"capacity": 2000000},
    num_sgd_iter=20,
)
```

---

## 第五部分：分布式训练

### 5.1 单机多GPU

```python
# 4×GPU配置
import ray
ray.init(num_gpus=4, num_cpus=32)

config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 100})

config.resources(num_gpus=0, num_cpus_for_driver=8)
config.rollouts(
    num_rollout_workers=24,
    num_envs_per_worker=4,
    rollout_fragment_length=1000,
)
config.training(train_batch_size=48000)
config.learners(
    num_learner_workers=4,
    num_gpus_per_learner_worker=1,
)

algo = config.build()
```

### 5.2 Ray集群训练

**启动集群**:
```bash
# Head节点 (192.168.1.100)
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Worker节点
ray start --address='192.168.1.100:6379'

# 查看集群
ray status
```

**提交训练**:
```python
import ray

# 连接集群
ray.init(address='auto')

config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 200})

# 集群资源配置
config.resources(num_gpus=8, num_cpus_for_driver=16)
config.rollouts(
    num_rollout_workers=64,
    num_envs_per_worker=4,
    rollout_fragment_length=2000,
    remote_worker_envs=True,
)
config.training(train_batch_size=256000)

algo = config.build()
for i in range(10000):
    result = algo.train()
```

### 5.3 资源分配策略

```python
# CPU密集型 (环境复杂)
config.resources(
    num_gpus=1,
    num_cpus_per_worker=2,  # 每worker 2 CPU
)

# GPU密集型 (模型大)
config.resources(
    num_gpus=4,
    num_cpus_per_worker=1,
    num_gpus_per_worker=0.1,  # Worker共享GPU
)

# 混合型
config.resources(
    num_gpus=2,
    num_cpus_per_worker=1.5,
)
config.learners(
    num_learner_workers=2,
    num_gpus_per_learner_worker=1,
)
```

---

## 第六部分：实战案例

### 6.1 本地开发 (Mac笔记本)

```python
config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 10})
config.resources(num_gpus=0, num_cpus_for_driver=1)
config.rollouts(num_rollout_workers=2, rollout_fragment_length=200)
config.training(train_batch_size=400, replay_buffer_config={"capacity": 10000})

# 预期: ~5分钟/100迭代, ~2GB内存
```

### 6.2 单GPU工作站

```python
config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 50})
config.resources(num_gpus=1, num_cpus_for_driver=2)
config.rollouts(num_rollout_workers=8, num_envs_per_worker=2, rollout_fragment_length=500)
config.training(train_batch_size=8000, replay_buffer_config={"capacity": 100000})

# 预期: ~2小时/1000迭代, GPU利用率70%, 8000样本/秒
```

### 6.3 多GPU服务器 (4×GPU)

```python
config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 100})
config.resources(num_gpus=4, num_cpus_for_driver=8)
config.rollouts(
    num_rollout_workers=32,
    num_envs_per_worker=4,
    rollout_fragment_length=1000,
    remote_worker_envs=True,
)
config.training(train_batch_size=32000)
config.learners(num_learner_workers=4, num_gpus_per_learner_worker=1)

# 预期: ~12小时/10000迭代, GPU利用率85%, 30000样本/秒
```

### 6.4 Ray集群 (10节点)

```python
ray.init(address='auto')

config = SACConfig()
config.environment(env="berth_allocation", env_config={"num_vessels": 200})
config.resources(num_gpus=8, num_cpus_for_driver=16)
config.rollouts(
    num_rollout_workers=64,
    num_envs_per_worker=4,
    rollout_fragment_length=2000,
    remote_worker_envs=True,
)
config.training(train_batch_size=64000)
config.learners(num_learner_workers=8, num_gpus_per_learner_worker=1)

# 预期: ~24小时/100000迭代, 集群GPU利用率80%, 100000样本/秒
```

### 6.5 性能对比

| 配置 | 硬件 | 时间(100艘船,5000迭代) | 加速比 |
|------|------|---------------------|--------|
| 单机CPU | 8核 | 48h | 1× |
| 单机GPU | 8核+1GPU | 18h | 2.7× |
| 单机4GPU | 32核+4GPU | 6h | 8× |
| 3节点集群 | 32核+4GPU×3 | 2.5h | 19× |
| 10节点集群 | 384核+48GPU | 1h | 48× |

---

## 附录A：快速命令参考

### 训练命令

```bash
# 基础训练
python rllib_train.py --algo SAC --num-vessels 50 --iterations 1000

# 自动优化
python rllib_train_advanced.py --auto-resources --optimize-for speed

# GPU训练
python rllib_train_advanced.py --gpus 2 --workers 16

# 集群训练
python rllib_train_advanced.py --distributed --auto-resources

# 性能分析
python rllib_train_advanced.py --profile --local
```

### Ray集群命令

```bash
# 启动head
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 启动worker
ray start --address='HEAD_IP:6379'

# 查看状态
ray status

# 停止Ray
ray stop
```

### 监控命令

```bash
# Ray Dashboard
open http://localhost:8265

# TensorBoard
tensorboard --logdir=~/ray_results --port=6006
```

---

## 附录B：性能调优清单

### 必做优化
- [x] 设置合理worker数量 (CPU核数-2)
- [x] 配置适当batch大小 (workers × envs × fragment)
- [x] 启用GPU加速 (如有)
- [x] 启用观测压缩
- [x] 定期保存检查点

### 推荐优化
- [ ] 调整学习率
- [ ] 优化replay buffer大小
- [ ] 使用混合精度训练
- [ ] 配置多GPU learner
- [ ] 优化环境代码

### 高级优化
- [ ] 自定义网络架构
- [ ] 实现自定义采样策略
- [ ] 使用异步训练(APPO)
- [ ] 实现模型压缩

---

## 附录C：故障排查

### 问题1: Worker连接超时
```python
# 增加worker数量或减少启动超时
config.rollouts(num_rollout_workers=8)  # 从16减到8
```

### 问题2: GPU内存不足
```python
# 减少batch大小
config.training(sgd_minibatch_size=128)  # 从512减到128
```

### 问题3: Object Store满
```bash
# 重启Ray,增加object store
ray stop
ray start --head --object-store-memory=100000000000
```

### 问题4: 奖励不收敛
```python
# 降低学习率,增加训练时间
config.training(
    optimization={"actor_learning_rate": 1e-4}  # 从3e-4降到1e-4
)
```

---

## 附录D：性能目标

| 规模 | 船舶数 | 吞吐量 | 训练时间(1000迭代) |
|------|--------|--------|------------------|
| 小规模 | 10 | 2K样本/秒 | <1小时 |
| 中规模 | 50 | 10K样本/秒 | <6小时 |
| 大规模 | 100 | 30K样本/秒 | <12小时 |
| 超大规模 | 200+ | 100K样本/秒 | <24小时 |

**GPU利用率目标**:
- 单GPU: 70-85%
- 多GPU (2-4): 75-90%
- 集群 (8+): 65-80%

---

**文档维护**: Duan
**最后更新**: 2025-10-28
**RLlib版本**: 2.50.1
