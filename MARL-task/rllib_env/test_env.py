"""测试RLlib泊位分配环境"""
import numpy as np
from berth_allocation_env import BerthAllocationMultiAgentEnv

def main():
    print("=" * 60)
    print("测试RLlib泊位分配环境")
    print("=" * 60)

    # 配置
    config = {
        'berth_length': 2000,
        'max_vessels': 5,  # 测试用小数量
        'planning_horizon': 168,
        'shore_power_enabled': True,
        'generation_mode': 'simple',
        'reward_weights': {
            'c1_base': 100.0,
            'c2_waiting': 10.0,
            'c3_emission': 0.01,
            'c4_shore_power': 50.0,
            'c5_utilization': 200.0,
            'c6_spacing': 30.0,
        }
    }

    # 创建环境
    print("\n1. 创建环境...")
    env = BerthAllocationMultiAgentEnv(config)
    print(f"   ✓ 环境创建成功")
    print(f"   观测空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")

    # 重置
    print("\n2. 重置环境...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ 环境重置成功")
    print(f"   智能体数量: {len(obs)}")
    print(f"   智能体IDs: {list(obs.keys())}")
    print(f"   观测示例 (vessel_0): shape={obs['vessel_0'].shape}")
    print(f"              values={obs['vessel_0'][:5]}...")  # 显示前5维

    # 运行几步
    print("\n3. 运行测试episode...")
    total_rewards = {agent_id: 0.0 for agent_id in obs.keys()}

    for step in range(10):  # 最多10步
        # 生成随机动作
        actions = {
            agent_id: env.action_space.sample()
            for agent_id in obs.keys()
            if agent_id not in env.allocated_agents
        }

        if not actions:  # 所有智能体都已分配
            break

        # 执行动作
        obs, rewards, dones, truncs, infos = env.step(actions)

        # 累计奖励
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0) + reward

        print(f"\n   步骤 {step + 1}:")
        print(f"     动作数量: {len(actions)}")
        print(f"     奖励: {', '.join([f'{k}={v:.1f}' for k, v in list(rewards.items())[:3]])}...")
        print(f"     已分配: {len(env.allocated_agents)}/{len(env.vessels)}")

        # 检查是否完成
        if dones.get('__all__', False):
            print(f"\n   ✓ Episode 完成!")
            break

    # 总结
    print("\n4. 测试总结:")
    print(f"   总步数: {step + 1}")
    print(f"   已分配船舶: {len(env.allocated_agents)}/{len(env.vessels)}")
    print(f"   总分配数: {len(env.allocations)}")

    if env.allocations:
        print("\n   分配示例:")
        for i, alloc in enumerate(env.allocations[:3]):  # 显示前3个
            print(f"     [{i}] 船舶{alloc['vessel_id']}: "
                  f"位置={alloc['position']:.1f}m, "
                  f"等待={alloc['waiting_time']:.1f}h, "
                  f"岸电={alloc['uses_shore_power']}")

    print(f"\n   平均奖励: {np.mean(list(total_rewards.values())):.2f}")

    # 清理
    env.close()

    print("\n" + "=" * 60)
    print("✅ 环境测试通过!")
    print("=" * 60)

if __name__ == "__main__":
    main()
