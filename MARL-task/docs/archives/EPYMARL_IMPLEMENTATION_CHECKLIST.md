# EPyMARL实施清单

## ✅ 准备工作（已完成）

- [x] 下载EPyMARL框架到项目目录
- [x] 分析EPyMARL架构和核心组件
- [x] 制定详细实施方案

---

## 📋 第一阶段：环境适配（Week 1-2）

### 任务1.1: 创建环境包装器
- [ ] 创建 `epymarl/src/envs/berth_allocation_wrapper.py`
- [ ] 实现 `BerthAllocationEnv` 类继承 `MultiAgentEnv`
- [ ] 实现 `reset()` 方法
- [ ] 实现 `step()` 方法
- [ ] 实现 `get_obs()` 和 `get_obs_agent()`
- [ ] 实现 `get_state()` (全局状态)
- [ ] 实现 `get_avail_actions()` (动作掩码)
- [ ] 实现 `get_env_info()` (环境信息)

### 任务1.2: 集成现有组件
- [ ] 复制 `environment/vessel.py` 到EPyMARL
- [ ] 复制 `environment/shore_power.py` 到EPyMARL
- [ ] 适配 `VesselGenerator` 用于环境重置
- [ ] 适配 `ShorePowerManager` 用于状态计算

### 任务1.3: 动作空间设计
- [ ] 决定：离散化 vs 连续动作
- [ ] 如选离散：实现动作映射函数
- [ ] 如选连续：扩展EPyMARL支持连续动作

### 任务1.4: 测试环境
- [ ] 创建 `test_berth_env.py`
- [ ] 测试环境重置
- [ ] 测试观测获取
- [ ] 测试状态获取
- [ ] 测试动作执行
- [ ] 验证奖励计算

---

## 📋 第二阶段：算法实现（Week 3-4）

### 任务2.1: MATD3学习器
- [ ] 创建 `epymarl/src/learners/matd3_learner.py`
- [ ] 基于 `maddpg_learner.py` 扩展
- [ ] 实现双Critic网络
- [ ] 实现延迟策略更新逻辑
- [ ] 实现目标策略平滑
- [ ] 添加探索噪声机制

### 任务2.2: 控制器
- [ ] 创建 `epymarl/src/controllers/matd3_controller.py`
- [ ] 实现连续动作选择
- [ ] 实现探索噪声添加
- [ ] 实现动作裁剪

### 任务2.3: 网络模块
- [ ] 创建 `epymarl/src/modules/agents/matd3_agent.py` (Actor)
- [ ] 创建 `epymarl/src/modules/critics/matd3_critic.py` (Critic)
- [ ] 定义网络架构 (MLP, 256 hidden)
- [ ] 测试前向传播

### 任务2.4: 注册组件
- [ ] 在 `learners/__init__.py` 注册MATD3Learner
- [ ] 在 `controllers/__init__.py` 注册MATD3Controller
- [ ] 在 `modules/agents/__init__.py` 注册MATD3Agent
- [ ] 在 `modules/critics/__init__.py` 注册MATD3Critic

---

## 📋 第三阶段：配置集成（Week 5）

### 任务3.1: 算法配置
- [ ] 创建 `epymarl/src/config/algs/matd3.yaml`
- [ ] 设置超参数（lr, batch_size等）
- [ ] 设置MATD3特有参数（policy_delay等）

### 任务3.2: 环境配置
- [ ] 创建 `epymarl/src/config/envs/berth_allocation.yaml`
- [ ] 配置环境参数
- [ ] 配置奖励权重
- [ ] 配置训练参数

### 任务3.3: 环境注册
- [ ] 在 `epymarl/src/envs/__init__.py` 注册环境
- [ ] 测试环境加载

### 任务3.4: 端到端测试
- [ ] 运行一个episode测试完整流程
- [ ] 检查数据流（obs -> action -> reward）
- [ ] 验证日志输出

---

## 📋 第四阶段：后端集成（Week 6）

### 任务4.1: EPyMARL运行器
- [ ] 创建 `backend/services/epymarl_runner.py`
- [ ] 实现 `run_algorithm()` 方法
- [ ] 实现参数转换逻辑
- [ ] 实现结果提取逻辑

### 任务4.2: API更新
- [ ] 修改 `backend/api/algorithm.py`
- [ ] 添加EPyMARL算法判断
- [ ] 路由到EPyMARLRunner
- [ ] 更新响应模型

### 任务4.3: 前端支持
- [ ] 在 `frontend/static/index.html` 添加新算法选项
  - QMIX
  - MAPPO
  - MADDPG
- [ ] 更新算法下拉菜单

### 任务4.4: 测试API
- [ ] 测试任务生成
- [ ] 测试MATD3算法运行
- [ ] 测试其他EPyMARL算法
- [ ] 验证性能指标正确性

---

## 📋 第五阶段：训练优化（Week 7）

### 任务5.1: 训练脚本
- [ ] 创建 `train_epymarl.py`
- [ ] 配置训练参数
- [ ] 设置日志路径
- [ ] 设置模型保存

### 任务5.2: 运行实验
- [ ] 训练MATD3模型（1M steps）
- [ ] 训练MADDPG模型（对比）
- [ ] 训练QMIX模型（对比）
- [ ] 记录训练曲线

### 任务5.3: 模型评估
- [ ] 评估训练好的模型
- [ ] 与Greedy/FCFS对比
- [ ] 生成性能报告
- [ ] 绘制对比图表

### 任务5.4: 文档更新
- [ ] 更新README添加EPyMARL说明
- [ ] 更新TROUBLESHOOTING
- [ ] 创建EPyMARL使用指南
- [ ] 添加训练示例

---

## 📋 额外任务（可选）

### 高级功能
- [ ] 集成Weights & Biases日志
- [ ] 实现超参数搜索
- [ ] 添加更多基线算法（QPLEX, QTRAN）
- [ ] 支持RNN网络（处理部分可观测）

### 实验分析
- [ ] Ablation study（消融实验）
- [ ] 奖励权重敏感性分析
- [ ] 船舶数量泛化性测试
- [ ] 岸电vs无岸电对比

### 论文相关
- [ ] 生成论文所需图表
- [ ] 记录实验配置
- [ ] 整理实验数据
- [ ] 准备补充材料

---

## 🎯 关键里程碑

- **Milestone 1** (Week 2): 环境wrapper可运行
- **Milestone 2** (Week 4): MATD3算法训练成功
- **Milestone 3** (Week 5): Web界面可调用EPyMARL算法
- **Milestone 4** (Week 7): 完整实验结果ready

---

## 📞 需要确认的问题

1. **动作空间**: 离散化（400动作）还是保持连续（3维）？
   - 建议：先离散化快速验证，后续扩展连续

2. **算法优先级**: 先实现哪些算法？
   - 建议：MATD3 > MADDPG > QMIX > MAPPO

3. **训练资源**: 是否有GPU可用？
   - CPU训练可行但较慢

4. **实验目标**: 侧重算法对比 or 实际部署？
   - 影响模型复杂度和训练时长

---

## 📝 当前状态

- ✅ EPyMARL已下载到 `MARL-task/epymarl/`
- ✅ 详细实施方案已完成
- ⏳ 等待确认后开始实施

**下一步**: 请告诉我是否开始实施？从哪个任务开始？
