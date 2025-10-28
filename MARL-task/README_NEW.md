# 🚢 多智能体强化学习泊位分配与岸电协同优化系统

**Multi-Agent Reinforcement Learning for Berth Allocation and Shore Power Coordination**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![RLlib](https://img.shields.io/badge/RLlib-2.50.1-green)](https://docs.ray.io/en/latest/rllib/)

---

## 📋 项目简介

使用**多智能体强化学习 (MARL)** 解决港口泊位分配与岸电协同优化问题:

- 🚢 **智能泊位分配** - 最小化船舶等待时间,提高泊位利用率
- ⚡ **岸电优化** - 优化岸电使用,降低碳排放
- 🎯 **多目标平衡** - 效率、环保、经济性协同优化
- 🤖 **先进算法** - 基于RLlib的SAC/PPO/TD3算法

**当前版本**: v2.0 | **框架**: RLlib 2.50.1 | **更新**: 2025-10-28

---

## ✨ 核心特性

### 多智能体环境
- ✅ 基于Gymnasium标准环境
- ✅ 17维连续观测空间
- ✅ 3维连续动作空间
- ✅ 动态智能体数量

### 强化学习算法
- 🚀 **SAC** (Soft Actor-Critic) - 推荐,高样本效率
- 🎯 **PPO** (Proximal Policy Optimization) - 稳定baseline
- ⚡ **TD3** (Twin Delayed DDPG) - 高精度控制

### 训练与部署
- ✅ 分布式训练 (Ray集群)
- ✅ GPU加速支持
- ✅ TensorBoard实时监控
- ✅ 自动超参数调优
- ✅ Web可视化界面

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd /Users/duan/mac-miclsirr/Sensors-models/MARL-task

# 创建环境
conda create -n marl-task python=3.8
conda activate marl-task

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "ray[rllib]==2.50.1"
pip install gymnasium pandas numpy pyyaml matplotlib seaborn tensorboard
```

### 2. 测试环境

```bash
# 测试RLlib环境
python rllib_env/test_env.py
```

### 3. 开始训练

```bash
# 本地快速测试 (10艘船, 100迭代, CPU)
python rllib_train.py --algo SAC --num-vessels 10 --iterations 100 --local

# GPU训练 (50艘船, 1000迭代)
python rllib_train.py --algo SAC --num-vessels 50 --iterations 1000 --gpus 1 --workers 8
```

### 4. 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir=./ray_results --port=6006

# 浏览器访问
open http://localhost:6006
```

---

## 📊 项目架构

```
MARL-task/
├── rllib_env/              # RLlib环境实现
│   ├── berth_allocation_env.py
│   └── test_env.py
├── rllib_train.py          # 训练脚本
│
├── environment/            # 基础环境模块
│   ├── berth_env.py
│   ├── vessel.py
│   └── shore_power.py
├── rewards/                # 奖励函数
│
├── backend/                # FastAPI后端
├── frontend/               # Web前端
│
├── config/                 # 配置文件
├── data/                   # 数据集
├── results/                # 训练结果
├── ray_results/            # RLlib检查点
│
├── docs/                   # 📚 完整文档
│   ├── guides/            # 使用指南
│   ├── references/        # 技术参考
│   ├── development/       # 开发文档
│   └── archives/          # 归档文档
│
└── archived/              # 旧框架归档
    ├── epymarl_backup_20251028/
    └── matd3_backup_20251028/
```

---

## 📚 文档导航

### 快速入门
- **[快速开始指南](docs/guides/QUICK_START.md)** - 5分钟上手
- **[RLlib框架指南](docs/guides/RLLIB_GUIDE.md)** - RLlib详细实现

### 技术参考
- **[环境设计](docs/references/ENVIRONMENT.md)** - 观测/动作/奖励详解
- **[算法详解](docs/references/ALGORITHMS.md)** - SAC/PPO/TD3原理
- **[问题排查](docs/references/TROUBLESHOOTING.md)** - 常见问题解决

### 归档与历史
- **[归档索引](docs/archives/ARCHIVE_INDEX.md)** - 旧框架代码说明
- **[框架对比](docs/archives/MARL_FRAMEWORKS_COMPARISON.md)** - 框架选择分析

**完整文档导航**: 参见 [docs/README.md](docs/README.md)

---

## 🎓 环境设计

### 观测空间 (17维)

| 维度 | 描述 | 范围 |
|------|------|------|
| 0-2 | 当前动作状态 | 位置, 等待时间, 岸电概率 |
| 3-5 | 船舶特征 | 长度, 到港时间, 优先级 |
| 6-10 | 岸电负载 | 5段岸电使用率 |
| 11-15 | 全局指标 | 利用率, 等待, 排放, 岸电率, 成功率 |
| 16 | 时间进度 | 归一化时间步 |

### 动作空间 (3维)

| 维度 | 含义 | 映射 |
|------|------|------|
| 0 | 泊位位置 | [-1,1] → [0, berth_length] |
| 1 | 等待时间 | [-1,1] → [0, max_waiting] |
| 2 | 岸电概率 | [-1,1] → [0, 1] |

### 奖励函数

```python
reward = c1 * base_reward          # 成功靠泊
       - c2 * waiting_penalty      # 等待时间
       - c3 * emission_penalty     # 碳排放
       + c4 * shore_power_bonus    # 岸电使用
       + c5 * utilization_reward   # 泊位利用率
       + c6 * spacing_reward       # 分散靠泊
```

---

## 🤖 支持算法

### SAC (推荐)

**特点**: 最大熵强化学习,高样本效率,适合连续动作

**训练**:
```bash
python rllib_train.py --algo SAC --num-vessels 50 --iterations 1000 --gpus 1
```

### PPO

**特点**: 稳定训练,易于调参,适合baseline

**训练**:
```bash
python rllib_train.py --algo PPO --num-vessels 30 --iterations 500 --workers 4
```

### TD3

**特点**: Twin Q-networks,高精度控制

**训练**:
```bash
python rllib_train.py --algo TD3 --num-vessels 40 --iterations 800 --gpus 1
```

**详细算法说明**: 参见 [算法详解](docs/references/ALGORITHMS.md)

---

## 📈 训练配置建议

| 场景 | 船舶数 | 迭代 | GPUs | Workers | 预计时间 |
|------|--------|------|------|---------|----------|
| 本地测试 | 10 | 100 | 0 | 2 | 30分钟 |
| 小规模实验 | 30 | 500 | 1 | 4 | 4小时 |
| 中规模实验 | 50 | 1000 | 1 | 8 | 8-12小时 |
| 大规模实验 | 100 | 5000 | 4 | 16 | 24-48小时 |

---

## 🌐 Web服务

### 启动后端

```bash
cd backend
python app.py
# 访问: http://localhost:8000/docs
```

### 启动前端

```bash
cd frontend
python -m http.server 3000
# 访问: http://localhost:3000
```

### API端点

- `POST /api/tasks` - 创建训练任务
- `GET /api/tasks/{task_id}` - 查询任务状态
- `POST /api/algorithms/run` - 运行算法
- `WS /ws/progress` - 实时进度推送

---

## 🔧 进阶功能

### 超参数调优

```python
from ray import tune

config = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "gamma": tune.uniform(0.95, 0.999),
}

tune.run("SAC", config=config, num_samples=20)
```

### 分布式训练

```bash
# 启动Ray集群
ray start --head

# 提交训练任务
python rllib_train.py --gpus 4 --workers 32 --distributed
```

### 模型导出

```python
# 导出ONNX
algo.export_policy_model(export_dir="./models", onnx=11)

# 加载检查点
algo = SAC.from_checkpoint("./ray_results/checkpoint_001000")
```

---

## 📦 归档框架

项目早期使用了其他框架,现已归档备份:

### EPyMARL框架
- **特点**: 学术标准框架,支持15+算法 (QMIX, MADDPG, MAPPO等)
- **位置**: `archived/epymarl_backup_20251028/`
- **用途**: 学术研究和算法benchmark

### 自研MATD3算法
- **特点**: 代码清晰,易于理解,适合教学
- **位置**: `archived/matd3_backup_20251028/`
- **用途**: 教学演示和快速原型

**详细说明**: 参见 [归档索引](docs/archives/ARCHIVE_INDEX.md)

---

## ❓ 常见问题

### 安装问题

**Q: 缺少ray模块?**
```bash
pip install "ray[rllib]==2.50.1"
```

**Q: GPU不可用?**
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装GPU版PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 训练问题

**Q: 奖励不收敛?**
- 降低学习率
- 增加训练时间
- 调整奖励权重

**Q: 显存不足?**
- 减少batch size
- 减少worker数量
- 使用CPU训练

**更多问题**: 参见 [问题排查手册](docs/references/TROUBLESHOOTING.md)

---

## 🤝 贡献

欢迎贡献! 重点方向:
- 新MARL算法集成
- 更复杂的港口场景
- 性能优化
- 文档完善

---

## 📝 引用

```bibtex
@article{berth-allocation-marl-2025,
  title={Multi-Agent Reinforcement Learning for Berth Allocation and Shore Power Coordination},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

---

## 📄 许可证

MIT License

---

## 👥 团队

- **开发者**: Duan
- **项目**: 泊位分配与岸电协同优化
- **框架**: RLlib 2.50.1
- **Python**: 3.8+

---

## 📞 获取帮助

- **完整文档**: [docs/README.md](docs/README.md)
- **快速开始**: [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)
- **问题排查**: [docs/references/TROUBLESHOOTING.md](docs/references/TROUBLESHOOTING.md)
- **归档代码**: [docs/archives/ARCHIVE_INDEX.md](docs/archives/ARCHIVE_INDEX.md)

---

## 🔄 版本历史

### v2.0 (2025-10-28)
- ✨ 迁移到RLlib框架
- ✨ 支持SAC/PPO/TD3算法
- ✨ 分布式训练支持
- 📦 归档EPyMARL和MATD3代码
- 📚 重组文档结构

### v1.0 (2024-10-25)
- 🎉 首次发布
- ✅ 自研MATD3实现
- ✅ Web可视化系统

---

**最后更新**: 2025-10-28 | **文档版本**: v2.0 | **状态**: 生产就绪 ✅
