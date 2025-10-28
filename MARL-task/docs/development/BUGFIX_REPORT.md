# Bug修复报告

**日期**: 2025-10-26
**版本**: v2.1

## 问题描述

用户在运行`./start_all.sh`启动系统后，执行算法时遇到500 Internal Server Error。

## 根本原因分析

### 问题1: 缺少`num_vessels`字段

**文件**: `rewards/reward_calculator.py`

**问题**: `calculate_metrics()`方法返回的metrics字典缺少`num_vessels`字段，但Pydantic响应模型`MetricsData`要求该字段为必填。

**错误**: FastAPI在序列化响应时验证失败，返回500错误。

### 问题2: MATD3 agent数量不匹配

**文件**: `backend/services/algorithm_runner.py`

**问题**: MATD3 agent使用`env_config['max_vessels']`初始化actor数量，但在realistic模式下，实际生成的vessel数量可能少于max_vessels，导致`self.actors[agent_id]`索引越界。

**错误**: `IndexError: list index out of range`

## 修复方案

### 修复1: 添加num_vessels字段

**文件**: `rewards/reward_calculator.py`

```python
# 第345-351行
return {
    'avg_waiting_time': avg_waiting_time,
    'total_emissions': total_emissions,
    'berth_utilization': berth_utilization,
    'shore_power_usage_rate': shore_power_usage_rate,
    'num_vessels': len(allocations)  # 新增
}

# 第305-311行（空列表情况）
return {
    'avg_waiting_time': 0.0,
    'total_emissions': 0.0,
    'berth_utilization': 0.0,
    'shore_power_usage_rate': 0.0,
    'num_vessels': 0  # 新增
}
```

### 修复2: 使用实际vessel数量

**文件**: `backend/services/algorithm_runner.py`

```python
# 第187-191行 (_run_matd3方法)
def _run_matd3(self, env: BerthAllocationEnv, model_path: str = None):
    """Run MATD3 algorithm."""
    # 使用实际vessel数量而非max_vessels
    num_agents = len(env.vessels)
    agent = self._load_agent(model_path, env.config, num_agents)
    # ...

# 第255-268行 (_load_agent方法签名修改)
def _load_agent(self, model_path: str, env_config: dict, num_agents: int = None) -> MATD3Agent:
    """Load or create MATD3 agent."""
    # ...

    # 使用提供的num_agents或回退到max_vessels
    if num_agents is None:
        num_agents = env_config['max_vessels']

    agent_config = {
        # ...
        'num_agents': num_agents,
        # ...
    }

# 第121-123行 (run_streaming方法)
# 同样修改streaming版本
num_agents = len(env.vessels)
agent = self._load_agent(model_path, env_config, num_agents)
```

## 测试结果

### 测试1: Simple模式 (8艘船)
```
✅ MATD3 - 成功
✅ Greedy - 成功
✅ FCFS - 成功
```

### 测试2: Realistic模式 (10艘船，实际生成数量可变)
```
✅ MATD3 - 成功 (6艘船分配)
✅ Greedy - 成功 (10艘船分配)
✅ FCFS - 成功 (10艘船分配)
```

## 影响范围

- ✅ 所有三个算法(MATD3, Greedy, FCFS)现在都能正常工作
- ✅ 同时支持simple和realistic生成模式
- ✅ API响应现在包含完整的metrics数据
- ✅ 修复不影响其他功能

## 后续建议

1. 添加单元测试验证metrics字段完整性
2. 为MATD3 agent添加动态vessel数量的集成测试
3. 考虑在API层面添加更详细的错误信息
