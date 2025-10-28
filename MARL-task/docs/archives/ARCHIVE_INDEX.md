# 📦 归档代码与文档索引

**归档时间**: 2025-10-28
**归档原因**: 项目迁移到RLlib框架

---

## 📂 归档内容概览

### 1. 归档代码

| 类别 | 位置 | 大小 | 状态 |
|------|------|------|------|
| **EPyMARL框架** | `../archived/epymarl_backup_20251028/` | ~50MB | ✅ 完整 |
| **自研MATD3** | `../archived/matd3_backup_20251028/` | ~5MB | ✅ 完整 |
| **相关文档** | `../archived/docs_backup_20251028/` | ~2MB | ✅ 完整 |

### 2. 归档文档

当前目录 (`docs/archives/`) 包含:
- `MARL_FRAMEWORKS_COMPARISON.md` - 框架详细对比
- `EPYMARL_INTEGRATION_PLAN.md` - EPyMARL集成计划
- `EPYMARL_IMPLEMENTATION_CHECKLIST.md` - EPyMARL实施清单
- `EPYMARL_VS_RLLIB_DETAILED.md` - EPyMARL与RLlib详细对比
- `RLLIB_INTEGRATION_SUMMARY.md` - RLlib集成总结
- `RLLIB_IMPLEMENTATION_PLAN.md` - RLlib实施计划
- `URGENT_FRAMEWORK_DECISION.md` - 框架决策说明

---

## 🔍 归档代码详情

### EPyMARL框架

**位置**: `../../archived/epymarl_backup_20251028/`

**简介**: Extended Python MARL学术标准框架

**支持算法** (15+):
- 价值分解: QMIX, VDN, QTRAN
- 策略梯度: MAPPO, IPPO, IA2C, MAA2C
- Actor-Critic: MADDPG, COMA, PAC, PAC-DCG
- Q-Learning: IQL

**目录结构**:
```
epymarl/
├── src/
│   ├── main.py              # 训练入口
│   ├── learners/            # 15+学习算法
│   ├── modules/             # 网络模块
│   ├── controllers/         # 动作控制器
│   ├── envs/                # 环境包装器
│   ├── runners/             # 训练运行器
│   └── config/              # 配置文件
└── requirements.txt
```

**恢复使用**:
```bash
# 1. 恢复代码
cp -r ../../archived/epymarl_backup_20251028/epymarl/ ../../

# 2. 安装依赖
cd ../../epymarl
pip install -r requirements.txt

# 3. 运行训练
cd src
python main.py --config=maddpg --env-config=berth_allocation
```

**适用场景**:
- ✅ 学术论文实验
- ✅ 算法benchmark对比
- ✅ 需要特定算法 (QMIX等)
- ✅ 离散动作空间任务

**详细说明**: 参见 `../../archived/epymarl_backup_20251028/README.md`

---

### 自研MATD3算法

**位置**: `../../archived/matd3_backup_20251028/`

**简介**: Multi-Agent Twin Delayed DDPG自主实现

**核心特性**:
- Twin Q-networks (双Q网络)
- Delayed policy updates (延迟策略更新)
- Target policy smoothing (目标策略平滑)
- 分维度探索噪声

**目录结构**:
```
matd3_backup_20251028/
├── agents/
│   ├── matd3.py             # MATD3核心算法 (~400行)
│   ├── networks.py          # Actor/Critic网络
│   └── replay_buffer.py     # 经验回放池
├── training/
│   ├── trainer.py           # 训练器
│   ├── evaluator.py         # 评估器
│   └── logger.py            # 日志系统
└── main.py                  # 训练入口
```

**恢复使用**:
```bash
# 1. 恢复代码
cp -r ../../archived/matd3_backup_20251028/agents/ ../../
cp -r ../../archived/matd3_backup_20251028/training/ ../../
cp ../../archived/matd3_backup_20251028/main.py ../../

# 2. 运行训练
cd ../../
python main.py --mode train --config config/default_config.yaml

# 3. 评估模型
python main.py --mode eval --checkpoint results/models/best_model.pth
```

**适用场景**:
- ✅ 算法学习和理解
- ✅ 教学演示
- ✅ 论文方法说明
- ✅ 快速原型验证
- ✅ 小规模实验 (<20 agents)

**超参数**:
```yaml
actor_learning_rate: 1e-4
critic_learning_rate: 1e-3
gamma: 0.99
tau: 0.005
policy_delay: 2
actor_hidden_dims: [256, 256]
critic_hidden_dims: [512, 512, 256]
```

**详细说明**: 参见 `../../archived/matd3_backup_20251028/README.md`

---

## 📊 框架对比

### 性能对比

| 特性 | RLlib (当前) | MATD3 (归档) | EPyMARL (归档) |
|------|-------------|--------------|----------------|
| **部署能力** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **训练速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **学术价值** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **代码清晰度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **可扩展性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **算法数量** | 10+ | 1 | 15+ |
| **GPU支持** | ✅ 原生 | ✅ 支持 | ✅ 支持 |
| **分布式训练** | ✅ 原生 | ❌ | ⚠️ 部分 |
| **文档质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 使用场景建议

**选择RLlib** (当前主力):
- ✅ 工业部署和生产环境
- ✅ 大规模训练 (50+ agents)
- ✅ GPU/分布式训练
- ✅ 快速实验迭代
- ✅ 需要多种算法对比

**选择MATD3** (归档-教学):
- ✅ 算法学习和理解
- ✅ 教学演示
- ✅ 论文方法说明
- ✅ 快速原型验证
- ✅ 小规模实验 (<20 agents)

**选择EPyMARL** (归档-学术):
- ✅ 学术论文实验
- ✅ 算法benchmark对比
- ✅ 需要特定算法 (QMIX等)
- ✅ 离散动作空间任务
- ✅ 学术会议提交

**详细对比**: 参见 [框架详细对比](MARL_FRAMEWORKS_COMPARISON.md)

---

## 📝 归档文档说明

### 框架对比类

| 文档 | 内容 | 用途 |
|------|------|------|
| `MARL_FRAMEWORKS_COMPARISON.md` | 3个框架全面对比 | 理解框架选择 |
| `EPYMARL_VS_RLLIB_DETAILED.md` | EPyMARL与RLlib详细对比 | 深入了解差异 |

### EPyMARL相关

| 文档 | 内容 | 用途 |
|------|------|------|
| `EPYMARL_INTEGRATION_PLAN.md` | EPyMARL集成计划 | 历史参考 |
| `EPYMARL_IMPLEMENTATION_CHECKLIST.md` | EPyMARL实施清单 | 实施参考 |

### RLlib迁移类

| 文档 | 内容 | 用途 |
|------|------|------|
| `RLLIB_INTEGRATION_SUMMARY.md` | RLlib集成总结 | 了解集成过程 |
| `RLLIB_IMPLEMENTATION_PLAN.md` | RLlib实施计划 | 实施参考 |
| `URGENT_FRAMEWORK_DECISION.md` | 框架决策说明 | 理解决策背景 |

---

## 🔄 迁移说明

### 为什么归档?

1. **EPyMARL**:
   - 学术框架,工业部署能力弱
   - 连续动作空间支持不完善
   - RLlib更适合生产环境

2. **MATD3**:
   - RLlib提供更完善的TD3/SAC实现
   - 缺少分布式训练支持
   - 保留作为教学参考

### 迁移时间线

| 时间 | 事件 |
|------|------|
| 2024-10-25 | 自研MATD3实现完成 |
| 2025-01-26 | 下载EPyMARL框架 |
| 2025-10-27 | RLlib集成完成 |
| 2025-10-28 | 归档旧框架代码 |

### 主要改进

**RLlib vs 旧框架**:
- ✅ 训练速度提升 3-5倍
- ✅ 支持分布式训练
- ✅ GPU利用率提高
- ✅ 更好的实验管理
- ✅ 更完善的文档和社区

---

## ⚠️ 注意事项

### 依赖兼容性

归档代码可能与当前环境有依赖冲突:

```bash
# 建议使用独立环境
conda create -n matd3-archive python=3.8
conda activate matd3-archive

# 或
conda create -n epymarl-archive python=3.8
conda activate epymarl-archive
```

### 配置文件

- 归档代码使用不同的配置格式
- 检查点和日志格式可能不兼容
- 需要适配环境参数

### API变化

- PyTorch/Gymnasium版本更新可能导致API不兼容
- 建议先在小规模数据上测试

---

## 📦 完整归档清单

执行以下命令查看归档清单:

```bash
cat ../../archived/ARCHIVE_MANIFEST.txt
```

或查看归档说明:

```bash
cat ../../archived/ARCHIVE_README.md
```

---

## 🛠️ 归档代码维护

- ✅ 代码已完整归档
- ✅ 文档已备份
- ⚠️ 不再主动维护
- ℹ️ 仅用于参考和特殊需求

---

## 📞 获取帮助

**使用归档代码遇到问题?**

1. 查看归档目录下的README文件
2. 参考框架对比文档
3. 查看迁移说明文档
4. 联系项目维护者

**推荐做法**: 优先使用当前RLlib框架实现

---

**归档维护**: Duan
**归档日期**: 2025-10-28
**当前框架**: RLlib 2.50.1
**文档版本**: v2.0
