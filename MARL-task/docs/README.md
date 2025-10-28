# 📚 项目文档导航

**多智能体强化学习泊位分配与岸电协同优化系统**

**更新时间**: 2025-10-28
**框架版本**: RLlib 2.50.1
**项目版本**: v2.0

---

## 🎯 快速导航

### 核心文档 (3个主要文档)

| 文档 | 内容 | 适合人群 |
|------|------|----------|
| **[RLlib完整指南](RLLIB_COMPLETE_GUIDE.md)** | RLlib框架、环境实现、训练配置、并行优化、分布式训练 | 所有开发者 ⭐⭐⭐⭐⭐ |
| **[算法设计](ALGORITHM_DESIGN.md)** | 问题建模、环境设计、奖励函数、算法实现、改进优化 | 算法研究者 ⭐⭐⭐⭐⭐ |
| **[系统架构](SYSTEM_ARCHITECTURE.md)** | 整体架构、核心模块、数据流程、部署架构、开发指南 | 系统工程师 ⭐⭐⭐⭐⭐ |

### 辅助文档

| 文档 | 内容 |
|------|------|
| **[问题排查](references/TROUBLESHOOTING.md)** | 常见问题和解决方案 |
| **[系统状态](development/SYSTEM_STATUS.md)** | 开发进度和日志 |

### 归档文档

| 文档 | 内容 |
|------|------|
| **[归档索引](archives/ARCHIVE_INDEX.md)** | EPyMARL和MATD3归档说明 |
| **[框架对比](archives/MARL_FRAMEWORKS_COMPARISON.md)** | 框架详细对比分析 |

---

## 📖 文档使用指南

### 我想要...

#### 🚀 快速上手训练
1. 阅读: [RLlib完整指南 - 第三部分](RLLIB_COMPLETE_GUIDE.md#第三部分训练配置)
2. 运行: `python rllib_train.py --algo SAC --local`

#### 🧠 理解RLlib框架
1. 阅读: [RLlib完整指南 - 第一部分](RLLIB_COMPLETE_GUIDE.md#第一部分rllib框架原理)
2. 学习: RLlib架构、训练循环、Worker机制

#### ⚡ 并行加速训练
1. 阅读: [RLlib完整指南 - 第四部分](RLLIB_COMPLETE_GUIDE.md#第四部分并行优化)
2. 运行: `python rllib_train_advanced.py --auto-resources --optimize-for speed`

#### 🌐 分布式集群训练
1. 阅读: [RLlib完整指南 - 第五部分](RLLIB_COMPLETE_GUIDE.md#第五部分分布式训练)
2. 配置: Ray集群 + 提交训练任务

#### 🎓 修改奖励函数
1. 阅读: [算法设计 - 第三部分](ALGORITHM_DESIGN.md#第三部分奖励函数)
2. 编辑: `rewards/reward_calculator.py`

#### 🔧 添加新算法
1. 阅读: [RLlib完整指南 - 第三部分](RLLIB_COMPLETE_GUIDE.md#第三部分训练配置)
2. 参考: SAC/PPO配置示例

#### 🏗️ 理解系统架构
1. 阅读: [系统架构 - 第一部分](SYSTEM_ARCHITECTURE.md#第一部分整体架构)
2. 查看: 项目目录结构

#### 🐛 解决问题
1. 查看: [问题排查手册](references/TROUBLESHOOTING.md)
2. 检查: [系统状态](development/SYSTEM_STATUS.md)

---

## 📚 文档详细说明

### 1. RLlib完整指南 (RLLIB_COMPLETE_GUIDE.md)

**文档长度**: ~8000字
**核心内容**:

#### 第一部分：RLlib框架原理
- RLlib架构概览 (7层架构)
- 训练循环详解 (伪代码)
- Worker架构说明

#### 第二部分：环境实现
- MultiAgentEnv基类实现
- 环境注册方法
- 三种多智能体策略模式

#### 第三部分：训练配置
- 基础配置模板
- SAC/PPO算法配置
- 完整训练脚本

#### 第四部分：并行优化
- 数据并行机制
- 自动资源优化
- GPU/内存优化

#### 第五部分：分布式训练
- 单机多GPU配置
- Ray集群训练
- 资源分配策略

#### 第六部分：实战案例
- 4个真实配置案例
- 性能对比数据

---

### 2. 算法设计 (ALGORITHM_DESIGN.md)

**文档长度**: ~6000字
**核心内容**:

#### 第一部分：问题建模
- 问题描述和目标
- MARL建模框架
- CTDE架构

#### 第二部分：环境设计
- 观测空间 (17维详解)
- 动作空间 (3维详解)
- 船舶生成 (非齐次泊松)
- 岸电管理 (5段管理)

#### 第三部分：奖励函数
- 多目标奖励设计
- 6个分项详解
- 权重配置

#### 第四部分：算法实现
- SAC算法 (推荐)
- PPO算法
- 算法对比

#### 第五部分：改进与优化
- 论文评审改进
- 权重敏感性分析
- 网络架构优化

---

### 3. 系统架构 (SYSTEM_ARCHITECTURE.md)

**文档长度**: ~7000字
**核心内容**:

#### 第一部分：整体架构
- 系统层次架构 (4层)
- 项目目录结构

#### 第二部分：核心模块
- 环境模块 (environment/)
- RLlib环境 (rllib_env/)
- 奖励模块 (rewards/)
- 后端服务 (backend/)
- 前端模块 (frontend/)

#### 第三部分：数据流程
- 训练数据流
- 推理数据流
- API请求流程

#### 第四部分：部署架构
- 本地开发部署
- 单机生产部署
- 集群部署架构

#### 第五部分：开发指南
- 环境搭建步骤
- 开发工作流
- 扩展指南

---

## 🎓 学习路径

### 初学者路径 (1-3天)

```
Day 1: 快速上手
  ↓
阅读 README.md → 了解项目
  ↓
阅读 RLlib完整指南 - 第三部分 → 训练配置
  ↓
运行 rllib_train.py → 第一次训练

Day 2: 理解环境
  ↓
阅读 算法设计 - 第二部分 → 环境设计
  ↓
阅读 算法设计 - 第三部分 → 奖励函数
  ↓
修改参数 → 重新训练

Day 3: 优化性能
  ↓
阅读 RLlib完整指南 - 第四部分 → 并行优化
  ↓
运行 rllib_train_advanced.py → 自动优化
```

### 研究者路径 (1-2周)

```
Week 1: 深入理解
  ↓
RLlib完整指南 - 第一部分 → RLlib框架
  ↓
RLlib完整指南 - 第二部分 → 环境实现
  ↓
算法设计 - 完整阅读 → 算法细节

Week 2: 实验优化
  ↓
修改奖励函数 → 消融实验
  ↓
调整网络架构 → 性能测试
  ↓
参数敏感性分析
```

### 工程师路径 (2-4周)

```
Week 1-2: 基础实现
  ↓
系统架构 - 第一、二部分 → 架构理解
  ↓
RLlib完整指南 → 训练实战

Week 3: 并行优化
  ↓
RLlib完整指南 - 第四、五部分 → 分布式
  ↓
部署Ray集群 → 大规模训练

Week 4: 生产部署
  ↓
系统架构 - 第四部分 → 部署架构
  ↓
API集成 → Web服务 → 上线
```

---

## 🔍 查找特定内容

### 环境相关
- **观测空间**: [算法设计 - 2.1](ALGORITHM_DESIGN.md#21-观测空间-17维)
- **动作空间**: [算法设计 - 2.2](ALGORITHM_DESIGN.md#22-动作空间-3维连续)
- **船舶生成**: [算法设计 - 2.3](ALGORITHM_DESIGN.md#23-船舶生成)
- **岸电管理**: [算法设计 - 2.4](ALGORITHM_DESIGN.md#24-岸电管理)

### RLlib相关
- **RLlib架构**: [RLlib完整指南 - 1.1](RLLIB_COMPLETE_GUIDE.md#11-rllib架构概览)
- **MultiAgentEnv**: [RLlib完整指南 - 2.1](RLLIB_COMPLETE_GUIDE.md#21-multiagentenv基类)
- **SAC配置**: [RLlib完整指南 - 3.2](RLLIB_COMPLETE_GUIDE.md#32-sac算法配置)
- **PPO配置**: [RLlib完整指南 - 3.3](RLLIB_COMPLETE_GUIDE.md#33-ppo算法配置)

### 优化相关
- **并行优化**: [RLlib完整指南 - 第四部分](RLLIB_COMPLETE_GUIDE.md#第四部分并行优化)
- **GPU优化**: [RLlib完整指南 - 4.3](RLLIB_COMPLETE_GUIDE.md#43-gpu优化)
- **分布式训练**: [RLlib完整指南 - 第五部分](RLLIB_COMPLETE_GUIDE.md#第五部分分布式训练)

### 系统相关
- **项目结构**: [系统架构 - 1.2](SYSTEM_ARCHITECTURE.md#12-项目目录结构)
- **核心模块**: [系统架构 - 第二部分](SYSTEM_ARCHITECTURE.md#第二部分核心模块)
- **部署方案**: [系统架构 - 第四部分](SYSTEM_ARCHITECTURE.md#第四部分部署架构)

---

## ⚡ 快速命令参考

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
```

### Ray集群
```bash
# Head节点
ray start --head --port=6379

# Worker节点
ray start --address='HEAD_IP:6379'

# 查看状态
ray status
```

### 监控
```bash
# TensorBoard
tensorboard --logdir=~/ray_results

# Ray Dashboard
open http://localhost:8265
```

---

## 📊 文档统计

**总文档数**: 8个
**核心文档**: 3个 (~21000字)
**辅助文档**: 2个
**归档文档**: 3个

**代码示例**: 80+
**配置案例**: 20+
**图表说明**: 15+

---

## 🤝 文档反馈

发现文档问题或有改进建议？

1. 查看 [贡献指南](development/CONTRIBUTING.md) (待创建)
2. 提交Issue
3. 联系维护者

---

## 📝 文档更新日志

### 2025-10-28 (v2.0)
- ✨ 合并RLlib相关文档为单一完整指南
- ✨ 创建算法设计统一文档
- ✨ 创建系统架构统一文档
- 🗑️ 删除冗余分散文档
- 📚 优化文档导航结构

### 2025-01-26 (v1.5)
- ✅ 添加EPyMARL集成文档
- ✅ 添加RLlib实施计划
- ✅ 添加框架对比分析

### 2024-10-25 (v1.0)
- 🎉 首次文档发布
- ✅ 项目README
- ✅ 系统状态文档

---

**文档维护**: Duan
**最后更新**: 2025-10-28
**文档版本**: v2.0
