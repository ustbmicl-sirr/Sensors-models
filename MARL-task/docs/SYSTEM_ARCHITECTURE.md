# 系统架构完整文档

**多智能体强化学习泊位分配系统 - 技术架构**

---

## 📋 目录

- [第一部分：整体架构](#第一部分整体架构)
- [第二部分：核心模块](#第二部分核心模块)
- [第三部分：数据流程](#第三部分数据流程)
- [第四部分：部署架构](#第四部分部署架构)
- [第五部分：开发指南](#第五部分开发指南)

---

## 第一部分：整体架构

### 1.1 系统层次架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户接口层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Web UI   │  │ REST API │  │ CLI      │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    业务逻辑层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ 任务管理     │  │ 算法调度     │  │ 结果缓存     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    训练引擎层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ RLlib训练    │  │ 环境管理     │  │ 模型管理     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    数据持久层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ 检查点存储   │  │ 日志存储     │  │ 结果存储     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 1.2 项目目录结构

```
MARL-task/
├── 📁 rllib_env/              # RLlib环境实现
│   ├── __init__.py
│   ├── berth_allocation_env.py
│   └── test_env.py
│
├── 📁 environment/            # 基础环境模块
│   ├── __init__.py
│   ├── berth_env.py          # Gymnasium环境
│   ├── vessel.py             # 船舶类和生成器
│   └── shore_power.py        # 岸电管理
│
├── 📁 rewards/                # 奖励函数模块
│   ├── __init__.py
│   └── reward_calculator.py
│
├── 📁 backend/                # FastAPI后端
│   ├── __init__.py
│   ├── app.py                # 主应用
│   ├── api/                  # API端点
│   │   ├── __init__.py
│   │   ├── task.py
│   │   ├── algorithm.py
│   │   └── websocket.py
│   ├── services/             # 业务逻辑
│   │   ├── __init__.py
│   │   ├── task_manager.py
│   │   ├── algorithm_runner.py
│   │   └── result_cache.py
│   ├── models/               # 数据模型
│   │   ├── __init__.py
│   │   ├── request.py
│   │   └── response.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
│
├── 📁 frontend/               # Web前端
│   └── static/
│       ├── index.html
│       ├── css/
│       ├── js/
│       └── assets/
│
├── 📁 config/                 # 配置文件
│   ├── default_config.yaml
│   └── logging_config.yaml
│
├── 📁 data/                   # 数据目录
│   ├── real_world/
│   └── synthetic/
│
├── 📁 results/                # 结果目录
│   ├── models/
│   ├── logs/
│   └── figures/
│
├── 📁 ray_results/            # RLlib训练结果
│
├── 📁 docs/                   # 文档
│   ├── RLLIB_COMPLETE_GUIDE.md
│   ├── ALGORITHM_DESIGN.md
│   └── SYSTEM_ARCHITECTURE.md
│
├── 📁 archived/               # 归档代码
│   ├── epymarl_backup_20251028/
│   └── matd3_backup_20251028/
│
├── 📄 rllib_train.py          # 基础训练脚本
├── 📄 rllib_train_advanced.py # 高级训练脚本
├── 📄 requirements.txt        # Python依赖
└── 📄 README.md              # 项目文档
```

---

## 第二部分：核心模块

### 2.1 环境模块 (environment/)

**berth_env.py - Gymnasium环境**:
```python
class BerthAllocationEnv(gym.Env):
    """泊位分配Gymnasium环境"""

    def __init__(self, config):
        # 初始化环境参数
        self.num_vessels = config.get("num_vessels", 10)
        self.berth_length = config.get("berth_length", 2000.0)

        # 定义空间
        self.observation_space = Box(...)
        self.action_space = Box(...)

        # 初始化状态
        self.vessels = []
        self.current_time = 0
        self.allocated_vessels = []

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 生成船舶
        self.vessels = self.vessel_generator.generate(self.num_vessels)

        # 重置状态
        self.current_time = 0
        self.allocated_vessels = []

        return self._get_obs(), {}

    def step(self, action):
        """执行一步"""
        # 解析动作
        position, waiting_time, shore_power_prob = self._parse_action(action)

        # 执行分配
        success = self._allocate_berth(position, waiting_time, shore_power_prob)

        # 计算奖励
        reward = self.reward_calculator.calculate_reward(...)

        # 更新状态
        self._update_state()

        # 检查终止
        done = self._check_done()

        return self._get_obs(), reward, done, False, {}
```

**vessel.py - 船舶管理**:
```python
class Vessel:
    """船舶数据类"""
    def __init__(self, vessel_id, length, arrival_time, priority, shore_power_capable):
        self.id = vessel_id
        self.length = length
        self.arrival_time = arrival_time
        self.priority = priority
        self.shore_power_capable = shore_power_capable

        # 运行时状态
        self.position = None
        self.waiting_time = 0
        self.berthing_time = None
        self.departure_time = None

class VesselGenerator:
    """船舶生成器"""
    def generate_realistic(self, num_vessels, horizon_days):
        """生成真实船舶序列"""
        vessels = []
        # 非齐次泊松过程
        for t in range(horizon_days * 24):
            rate = self._get_arrival_rate(t)
            num_arrivals = np.random.poisson(rate)
            for _ in range(num_arrivals):
                vessel = self._create_vessel(t)
                vessels.append(vessel)
        return vessels[:num_vessels]
```

**shore_power.py - 岸电管理**:
```python
class ShorePowerManager:
    """岸电管理系统"""
    def __init__(self, num_segments=5):
        self.num_segments = num_segments
        self.capacities = [500] * num_segments
        self.loads = [0] * num_segments

    def allocate(self, vessel, position, shore_power_prob):
        """分配岸电"""
        # 确定占用段
        start_seg, end_seg = self._get_segments(position, vessel.length)

        # 检查容量
        for seg in range(start_seg, end_seg + 1):
            required = vessel.power_demand * shore_power_prob
            if self.loads[seg] + required > self.capacities[seg]:
                return False, 0

        # 分配成功
        allocated_power = 0
        for seg in range(start_seg, end_seg + 1):
            power = vessel.power_demand * shore_power_prob
            self.loads[seg] += power
            allocated_power += power

        return True, allocated_power
```

### 2.2 RLlib环境模块 (rllib_env/)

**berth_allocation_env.py - MultiAgentEnv**:
```python
class BerthAllocationMultiAgentEnv(MultiAgentEnv):
    """RLlib多智能体环境包装"""

    def __init__(self, env_config):
        super().__init__()

        # 创建基础环境
        from environment.berth_env import BerthAllocationEnv
        self.base_env = BerthAllocationEnv(env_config)

        # 定义空间
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        self._agent_ids = set()

    def reset(self, *, seed=None, options=None):
        """重置环境"""
        obs, info = self.base_env.reset(seed=seed)

        # 生成智能体ID
        self._agent_ids = {f"vessel_{i}" for i in range(self.base_env.num_vessels)}

        # 多智能体观测
        observations = {agent_id: obs for agent_id in self._agent_ids}
        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, infos

    def step(self, action_dict):
        """执行一步"""
        # 聚合动作
        actions = np.array(list(action_dict.values()))
        aggregated_action = np.mean(actions, axis=0)

        # 执行
        obs, reward, terminated, truncated, info = self.base_env.step(aggregated_action)

        # 多智能体返回
        observations = {agent_id: obs for agent_id in self._agent_ids}
        rewards = {agent_id: reward for agent_id in self._agent_ids}
        terminateds = {agent_id: terminated for agent_id in self._agent_ids}
        truncateds = {agent_id: truncated for agent_id in self._agent_ids}
        infos = {agent_id: info for agent_id in self._agent_ids}

        # 全局终止
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        return observations, rewards, terminateds, truncateds, infos
```

### 2.3 奖励模块 (rewards/)

**reward_calculator.py**:
```python
class RewardCalculator:
    """奖励计算器"""

    def __init__(self, weights):
        self.weights = weights

    def calculate_reward(self, vessel, action, state):
        """计算奖励"""
        # 分项计算
        r1 = self._base_reward(vessel)
        r2 = self._waiting_penalty(vessel)
        r3 = self._emission_penalty(vessel, action)
        r4 = self._shore_power_bonus(vessel, action)
        r5 = self._utilization_reward(state)
        r6 = self._spacing_reward(vessel, action, state)

        # 加权求和
        total = (
            self.weights['c1'] * r1
            - self.weights['c2'] * r2
            - self.weights['c3'] * r3
            + self.weights['c4'] * r4
            + self.weights['c5'] * r5
            + self.weights['c6'] * r6
        )

        return total
```

### 2.4 后端服务模块 (backend/)

**app.py - FastAPI主应用**:
```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MARL Berth Allocation API")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# 注册路由
from api import task, algorithm, websocket
app.include_router(task.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(algorithm.router, prefix="/api/algorithms", tags=["algorithms"])

@app.get("/")
def root():
    return {"message": "MARL Berth Allocation System API"}
```

**services/algorithm_runner.py**:
```python
class AlgorithmRunner:
    """算法运行器"""

    def __init__(self):
        self.running_tasks = {}

    async def run_rllib_training(self, task_id, config):
        """运行RLlib训练"""
        import ray
        from ray.rllib.algorithms.sac import SACConfig

        # 初始化Ray
        ray.init(ignore_reinit_error=True)

        # 配置
        rllib_config = SACConfig()
        rllib_config.environment(env="berth_allocation", env_config=config)

        # 训练
        algo = rllib_config.build()
        for i in range(config['num_iterations']):
            result = algo.train()

            # 更新进度
            await self._update_progress(task_id, i, result)

        # 保存模型
        checkpoint = algo.save()

        algo.stop()
        ray.shutdown()

        return checkpoint
```

**services/task_manager.py**:
```python
class TaskManager:
    """任务管理器"""

    def __init__(self):
        self.tasks = {}

    def create_task(self, task_config):
        """创建任务"""
        task_id = str(uuid.uuid4())

        task = {
            'id': task_id,
            'status': 'pending',
            'config': task_config,
            'created_at': datetime.now(),
            'progress': 0,
            'result': None,
        }

        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id):
        """获取任务"""
        return self.tasks.get(task_id)

    def update_task(self, task_id, updates):
        """更新任务"""
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)
```

### 2.5 前端模块 (frontend/)

**index.html - 主页面**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>MARL Berth Allocation</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div id="app">
        <!-- 参数配置面板 -->
        <div class="config-panel">
            <h2>配置参数</h2>
            <input id="num-vessels" type="number" placeholder="船舶数量">
            <input id="planning-horizon" type="number" placeholder="规划周期(天)">
            <button onclick="createTask()">生成任务</button>
        </div>

        <!-- 算法选择面板 -->
        <div class="algorithm-panel">
            <h2>选择算法</h2>
            <select id="algorithm">
                <option value="SAC">SAC</option>
                <option value="PPO">PPO</option>
                <option value="Greedy">Greedy</option>
            </select>
            <button onclick="runAlgorithm()">运行算法</button>
        </div>

        <!-- 可视化面板 -->
        <div class="visualization-panel">
            <canvas id="berth-canvas"></canvas>
            <div id="metrics"></div>
        </div>
    </div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

**app.js - 前端逻辑**:
```javascript
// 创建任务
async function createTask() {
    const numVessels = document.getElementById('num-vessels').value;
    const planningHorizon = document.getElementById('planning-horizon').value;

    const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            num_vessels: parseInt(numVessels),
            planning_horizon_days: parseInt(planningHorizon),
        })
    });

    const data = await response.json();
    window.currentTaskId = data.task_id;
    console.log('Task created:', data.task_id);
}

// 运行算法
async function runAlgorithm() {
    const algorithm = document.getElementById('algorithm').value;

    const response = await fetch('/api/algorithms/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            task_id: window.currentTaskId,
            algorithm: algorithm,
        })
    });

    const data = await response.json();
    console.log('Algorithm started:', data);

    // WebSocket监听进度
    connectWebSocket(data.task_id);
}

// WebSocket连接
function connectWebSocket(taskId) {
    const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);

    ws.onmessage = function(event) {
        const progress = JSON.parse(event.data);
        updateUI(progress);
    };
}
```

---

## 第三部分：数据流程

### 3.1 训练数据流

```
用户输入
   ↓
配置参数 {num_vessels, horizon, ...}
   ↓
生成船舶序列 (VesselGenerator)
   ↓
初始化环境 (BerthAllocationEnv)
   ↓
RLlib训练循环:
   ├─ Worker采样 → 环境交互 → 收集样本
   ├─ 聚合样本 → Replay Buffer
   ├─ 采样Batch → GPU训练 → 更新策略
   └─ 评估 → 记录指标
   ↓
保存检查点
   ↓
返回结果 {reward, metrics, checkpoint}
```

### 3.2 推理数据流

```
加载检查点 (Checkpoint)
   ↓
初始化环境
   ↓
输入船舶数据
   ↓
循环:
   ├─ 获取观测 (Observation)
   ├─ 策略推理 (Policy Inference)
   ├─ 输出动作 (Action)
   └─ 执行动作 → 更新环境
   ↓
输出分配方案
   ↓
可视化结果
```

### 3.3 API请求流程

```
HTTP请求 → FastAPI路由
   ↓
请求验证 (Pydantic Model)
   ↓
业务逻辑 (Service Layer)
   ├─ TaskManager: 任务管理
   ├─ AlgorithmRunner: 算法运行
   └─ ResultCache: 结果缓存
   ↓
数据持久化
   ├─ 检查点存储
   ├─ 日志记录
   └─ 结果保存
   ↓
HTTP响应 → 前端
   ↓
WebSocket实时推送 (可选)
```

---

## 第四部分：部署架构

### 4.1 本地开发部署

```
┌─────────────────────────────────┐
│         开发机器                 │
│  ┌──────────────────────────┐  │
│  │  Python环境 (Conda)      │  │
│  │  - ray[rllib]==2.50.1    │  │
│  │  - torch>=2.0            │  │
│  │  - fastapi               │  │
│  └──────────────────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  训练进程                │  │
│  │  python rllib_train.py   │  │
│  └──────────────────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  后端服务 (Port 8000)    │  │
│  │  uvicorn app:app         │  │
│  └──────────────────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  前端服务 (Port 3000)    │  │
│  │  python -m http.server   │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

### 4.2 单机生产部署

```
┌─────────────────────────────────────────┐
│           生产服务器                     │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │  Docker Container                │  │
│  │  ┌────────────────────────────┐ │  │
│  │  │  Ray Head                  │ │  │
│  │  │  - 训练引擎               │ │  │
│  │  │  - GPU支持                │ │  │
│  │  └────────────────────────────┘ │  │
│  │                                  │  │
│  │  ┌────────────────────────────┐ │  │
│  │  │  FastAPI + Uvicorn         │ │  │
│  │  │  - REST API               │ │  │
│  │  │  - WebSocket              │ │  │
│  │  └────────────────────────────┘ │  │
│  │                                  │  │
│  │  ┌────────────────────────────┐ │  │
│  │  │  Nginx                     │ │  │
│  │  │  - 静态文件服务           │ │  │
│  │  │  - 反向代理               │ │  │
│  │  └────────────────────────────┘ │  │
│  └──────────────────────────────────┘  │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │  数据卷                          │  │
│  │  - /data/models                  │  │
│  │  - /data/logs                    │  │
│  │  - /data/results                 │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 4.3 集群部署

```
┌──────────────────────────────────────────────────┐
│                Ray集群架构                        │
└──────────────────────────────────────────────────┘

┌────────────────┐
│  Load Balancer │
│   (Nginx)      │
└────────┬───────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│ API1 │  │ API2 │  ← FastAPI实例 (多副本)
└──────┘  └──────┘
    │         │
    └────┬────┘
         │
┌────────▼─────────┐
│  Ray Head Node   │  ← 调度器 + Dashboard
│  - 16核 + 2GPU   │
└────────┬─────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
│Worker│  │Worker│  │Worker│  │Worker│  ← Worker节点
│8核+1G│  │8核+1G│  │8核+1G│  │8核+1G│
└──────┘  └─────┘  └─────┘  └─────┘

┌──────────────────────────────────────┐
│      共享存储 (NFS/S3)               │
│  - 模型检查点                        │
│  - 训练日志                          │
│  - 结果数据                          │
└──────────────────────────────────────┘
```

---

## 第五部分：开发指南

### 5.1 环境搭建

```bash
# 1. 克隆项目
git clone <repository>
cd MARL-task

# 2. 创建Conda环境
conda create -n marl-task python=3.8
conda activate marl-task

# 3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "ray[rllib]==2.50.1"
pip install gymnasium pandas numpy pyyaml matplotlib seaborn tensorboard
pip install fastapi uvicorn websockets

# 4. 验证安装
python -c "import ray; import torch; print('Ray:', ray.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 5.2 开发工作流

```bash
# 1. 测试环境
python rllib_env/test_env.py

# 2. 本地训练测试
python rllib_train.py --algo SAC --local --iterations 10

# 3. 启动后端
cd backend
uvicorn app:app --reload --port 8000

# 4. 启动前端
cd frontend
python -m http.server 3000

# 5. 访问
open http://localhost:3000
```

### 5.3 添加新算法

```python
# 1. 在rllib_train.py中添加算法配置
from ray.rllib.algorithms.ddpg import DDPGConfig

def create_ddpg_config(env_config):
    config = DDPGConfig()
    config.environment(env="berth_allocation", env_config=env_config)
    # ... 配置参数
    return config

# 2. 注册到算法字典
ALGORITHM_CONFIGS = {
    "SAC": create_sac_config,
    "PPO": create_ppo_config,
    "DDPG": create_ddpg_config,  # 新增
}
```

### 5.4 自定义网络

```python
# 1. 定义网络
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

class MyCustomNetwork(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 定义网络层
        self.net = nn.Sequential(
            nn.Linear(17, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)

# 2. 注册网络
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("my_network", MyCustomNetwork)

# 3. 使用网络
config.model = {
    "custom_model": "my_network",
    "custom_model_config": {},
}
```

### 5.5 扩展API端点

```python
# backend/api/custom.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/custom/endpoint")
async def custom_endpoint(data: dict):
    # 自定义逻辑
    result = process_data(data)
    return {"result": result}

# backend/app.py
from api import custom
app.include_router(custom.router, prefix="/api/custom")
```

---

## 附录A：技术栈

| 层次 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **框架** | Ray RLlib | 2.50.1 | 强化学习训练 |
| **深度学习** | PyTorch | 2.0+ | 神经网络 |
| **环境** | Gymnasium | 0.28+ | 环境标准 |
| **后端** | FastAPI | latest | REST API |
| **前端** | HTML/JS | - | Web UI |
| **可视化** | TensorBoard | 2.12+ | 训练监控 |
| **数据** | NumPy/Pandas | latest | 数据处理 |
| **配置** | PyYAML | latest | 配置管理 |

---

## 附录B：性能指标

| 模块 | 性能指标 | 目标值 |
|------|---------|--------|
| **环境** | step()耗时 | <5ms |
| **推理** | 单次推理 | <10ms |
| **API** | 响应时间 | <100ms |
| **训练** | 吞吐量 | >10K样本/秒 |
| **内存** | 峰值占用 | <8GB (50船) |

---

## 附录C：常用命令

```bash
# 训练
python rllib_train.py --algo SAC --num-vessels 50 --iterations 1000

# 启动服务
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Ray集群
ray start --head
ray status
ray stop

# 监控
tensorboard --logdir=./ray_results
open http://localhost:8265  # Ray Dashboard
```

---

**文档维护**: Duan
**最后更新**: 2025-10-28
**版本**: v2.0
