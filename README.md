# Sensors-models

这个主要是总结可以发的跟硬件有关的论文,特别是nature sensor这个期刊！

## 📁 项目结构

### 主项目

#### MARL-task/
多智能体强化学习（MARL）在连续泊位分配与岸电协同优化中的应用

- **框架**: RLlib 2.50.1
- **算法**: SAC, PPO, TD3
- **特性**: CTDE架构、连续动作空间、多目标优化
- **文档**: [MARL-task/docs/README.md](MARL-task/docs/README.md)

### 子模块

#### modelGrow/ (子模块)
自动网络增长与结构重参数化 - 模型优化

- **仓库**: https://github.com/ustbmicl-sirr/modelGrow
- **文档**: [SUBMODULES.md](SUBMODULES.md)

## 🚀 快速开始

### 克隆项目（包含子模块）

```bash
# 推荐：克隆时同时获取子模块
git clone --recurse-submodules https://github.com/ustbmicl-sirr/Sensors-models.git

# 或者分步克隆
git clone https://github.com/ustbmicl-sirr/Sensors-models.git
cd Sensors-models
git submodule init
git submodule update
```

### 使用MARL-task

```bash
cd MARL-task

# 检查环境
./manage.sh check

# 启动训练
./manage.sh train

# 启动所有服务
./manage.sh start
```

详细文档请查看：[MARL-task/README.md](MARL-task/README.md)

## 📚 文档导航

- **项目说明**: 本文档
- **子模块管理**: [SUBMODULES.md](SUBMODULES.md)
- **MARL项目文档**: [MARL-task/docs/README.md](MARL-task/docs/README.md)
- **RLlib完整指南**: [MARL-task/docs/RLLIB_COMPLETE_GUIDE.md](MARL-task/docs/RLLIB_COMPLETE_GUIDE.md)
- **算法设计**: [MARL-task/docs/ALGORITHM_DESIGN.md](MARL-task/docs/ALGORITHM_DESIGN.md)
- **系统架构**: [MARL-task/docs/SYSTEM_ARCHITECTURE.md](MARL-task/docs/SYSTEM_ARCHITECTURE.md)

## 🔗 相关链接

- **主仓库**: https://github.com/ustbmicl-sirr/Sensors-models
- **modelGrow子模块**: https://github.com/ustbmicl-sirr/modelGrow

---

**更新时间**: 2025-10-28
