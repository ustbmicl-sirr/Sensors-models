# Sensors-models

这个主要是总结可以发的跟硬件有关的论文,特别是nature sensor这个期刊！

## 📁 项目结构

本项目采用 Git Submodule 管理多个独立研究模块。

### 子模块

#### 1. MARL-tasks/ (子模块)
多智能体强化学习在连续泊位分配与岸电协同优化中的应用

- **仓库**: https://github.com/ustbmicl-sirr/MARL-tasks.git
- **算法**: MATD3, SAC, PPO, TD3
- **框架**: RLlib 2.50.1
- **特性**: CTDE架构、连续动作空间、多目标优化
- **应用**: 自动化集装箱码头泊位分配、岸电协同优化
- **文档**: [MARL-tasks/README.md](MARL-tasks/README.md)

#### 2. modelGrow/ (子模块)
自动网络增长与结构重参数化 - 模型优化

- **仓库**: https://github.com/ustbmicl-sirr/modelGrow.git
- **特性**: 自动网络增长、结构重参数化
- **应用**: 深度学习模型优化

#### 3. modelST/ (子模块)
模型结构重参数化

- **仓库**: https://github.com/ustbmicl-sirr/modelST.git
- **特性**: 结构重参数化技术
- **应用**: 模型压缩与加速

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

### 使用MARL-tasks

```bash
cd MARL-tasks

# 检查环境
./manage.sh check

# 启动训练
./manage.sh train

# 启动所有服务
./manage.sh start
```

详细文档请查看：[MARL-tasks/README.md](MARL-tasks/README.md)

## 📚 文档导航

- **项目说明**: 本文档
- **子模块管理**: [SUBMODULES.md](SUBMODULES.md)
- **MARL项目文档**: [MARL-tasks/README.md](MARL-tasks/README.md)
- **MARL文档目录**: [MARL-tasks/docs/README.md](MARL-tasks/docs/README.md)

## 🔗 相关链接

- **主仓库**: https://github.com/ustbmicl-sirr/Sensors-models
- **MARL-tasks子模块**: https://github.com/ustbmicl-sirr/MARL-tasks
- **modelGrow子模块**: https://github.com/ustbmicl-sirr/modelGrow
- **modelST子模块**: https://github.com/ustbmicl-sirr/modelST

---

**更新时间**: 2025-11-06
