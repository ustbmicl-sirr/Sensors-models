#!/usr/bin/env python3
"""
System integration test for MATD3 berth allocation.

Tests all components to ensure they work together correctly.
"""

import numpy as np
import torch
import yaml


def test_environment():
    """Test environment module."""
    print("\n" + "="*60)
    print("Testing Environment Module")
    print("="*60)

    from environment import Vessel, VesselGenerator, ShorePowerManager, BerthAllocationEnv

    # Test Vessel
    print("\n1. Testing Vessel class...")
    vessel = Vessel(
        id=0,
        length=200.0,
        draft=10.0,
        arrival_time=5.0,
        operation_time=10.0,
        priority=1,
        can_use_shore_power=True,
        power_requirement=500.0
    )
    print(f"   Created: {vessel}")
    assert vessel.length == 200.0
    print("   ✓ Vessel class OK")

    # Test VesselGenerator
    print("\n2. Testing VesselGenerator...")
    config = {
        'peak_hours': [6, 12, 18],
        'peak_rate': 2.0,
        'size_distribution': [0.3, 0.5, 0.2],
        'shore_power_ratio': 0.6
    }
    generator = VesselGenerator(config, seed=42)
    vessels = generator.generate_realistic(10, 7)
    print(f"   Generated {len(vessels)} vessels")
    print(f"   First vessel: {vessels[0]}")
    assert len(vessels) == 10
    print("   ✓ VesselGenerator OK")

    # Test ShorePowerManager
    print("\n3. Testing ShorePowerManager...")
    shore_config = {
        'segments': [
            {'start': 0, 'end': 400, 'capacity': 5000},
            {'start': 400, 'end': 800, 'capacity': 5000},
        ],
        'emission_factor_ship': 2500,
        'emission_factor_shore': 800
    }
    shore_power = ShorePowerManager(shore_config)
    print(f"   {shore_power}")

    # Test allocation
    available, segments = shore_power.check_availability(100, 200, 1000)
    print(f"   Shore power available: {available}, segments: {segments}")
    assert available == True
    print("   ✓ ShorePowerManager OK")

    # Test BerthAllocationEnv
    print("\n4. Testing BerthAllocationEnv...")
    env_config = {
        'berth_length': 2000,
        'planning_horizon': 7,
        'max_vessels': 5,
        'safe_distance': 20,
        'max_wait_time': 48.0,
        'shore_power': shore_config,
        'rewards': {f'c{i}': 1.0 for i in range(1, 9)},
        'vessel_generation': config,
        'seed': 42
    }
    env = BerthAllocationEnv(env_config)
    print(f"   Berth length: {env.berth_length}m")
    print(f"   Max vessels: {env.max_vessels}")

    observations, info = env.reset()
    print(f"   Reset: {len(observations)} agents, {info['num_vessels']} vessels")

    # Test step
    actions = {i: np.array([0.5, 0.1, 0.5]) for i in observations.keys()}
    next_obs, rewards, done, truncated, infos = env.step(actions)
    print(f"   Step: rewards={list(rewards.values())[:3]}")

    assert env.observation_space.shape == (17,)
    assert env.action_space.shape == (3,)
    print("   ✓ BerthAllocationEnv OK")

    print("\n✅ Environment module tests passed!")


def test_agents():
    """Test agent module."""
    print("\n" + "="*60)
    print("Testing Agent Module")
    print("="*60)

    from agents import Actor, Critic, ReplayBuffer, MATD3Agent

    # Parameters
    obs_dim = 17
    action_dim = 3
    num_agents = 5
    batch_size = 32

    # Test Actor
    print("\n1. Testing Actor...")
    actor = Actor(obs_dim, action_dim)
    obs = torch.randn(batch_size, obs_dim)
    action = actor(obs)
    assert action.shape == (batch_size, action_dim)
    assert torch.all((action >= 0) & (action <= 1))
    print(f"   Output shape: {action.shape}")
    print(f"   Range: [{action.min():.3f}, {action.max():.3f}]")
    print("   ✓ Actor OK")

    # Test Critic
    print("\n2. Testing Critic...")
    global_state_dim = num_agents * obs_dim
    critic = Critic(global_state_dim, num_agents, action_dim)
    global_state = torch.randn(batch_size, global_state_dim)
    all_actions = torch.randn(batch_size, num_agents * action_dim)
    q_value = critic(global_state, all_actions)
    assert q_value.shape == (batch_size, 1)
    print(f"   Output shape: {q_value.shape}")
    print("   ✓ Critic OK")

    # Test ReplayBuffer
    print("\n3. Testing ReplayBuffer...")
    buffer = ReplayBuffer(1000, obs_dim, action_dim, num_agents)

    for _ in range(100):
        obs_dict = {i: np.random.randn(obs_dim).astype(np.float32)
                   for i in range(num_agents)}
        act_dict = {i: np.random.rand(action_dim).astype(np.float32)
                   for i in range(num_agents)}
        rew_dict = {i: np.random.randn() for i in range(num_agents)}
        next_obs_dict = {i: np.random.randn(obs_dim).astype(np.float32)
                        for i in range(num_agents)}
        global_s = np.random.randn(global_state_dim).astype(np.float32)
        next_global_s = np.random.randn(global_state_dim).astype(np.float32)

        buffer.add(obs_dict, act_dict, rew_dict, next_obs_dict,
                  global_s, next_global_s, False)

    batch = buffer.sample(batch_size)
    assert batch['observations'].shape == (batch_size, num_agents, obs_dim)
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Batch shape: {batch['observations'].shape}")
    print("   ✓ ReplayBuffer OK")

    # Test MATD3Agent
    print("\n4. Testing MATD3Agent...")
    config = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'num_agents': num_agents,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'policy_delay': 2,
        'device': 'cpu'
    }
    agent = MATD3Agent(config)

    # Test action selection
    observations = {i: np.random.randn(obs_dim).astype(np.float32)
                   for i in range(num_agents)}
    actions = agent.select_action(observations, explore=True)
    assert len(actions) == num_agents
    print(f"   Action shape: {actions[0].shape}")
    print(f"   Action range: [{actions[0].min():.3f}, {actions[0].max():.3f}]")

    # Test update
    stats = agent.update(buffer, batch_size)
    assert 'critic_loss' in stats
    print(f"   Update stats: {list(stats.keys())}")
    print("   ✓ MATD3Agent OK")

    print("\n✅ Agent module tests passed!")


def test_rewards():
    """Test reward module."""
    print("\n" + "="*60)
    print("Testing Reward Module")
    print("="*60)

    from rewards import RewardCalculator
    from environment import Vessel

    config = {
        'rewards': {f'c{i}': 1.0 for i in range(1, 9)},
        'berth_length': 2000,
        'planning_horizon': 7,
        'max_wait_time': 48.0,
        'safe_distance': 20.0,
        'shore_power': {
            'emission_factor_ship': 2500,
            'emission_factor_shore': 800
        }
    }

    calculator = RewardCalculator(config)

    vessel = Vessel(
        id=0,
        length=200.0,
        draft=10.0,
        arrival_time=5.0,
        operation_time=10.0,
        priority=1,
        can_use_shore_power=True,
        power_requirement=500.0
    )

    allocation = {
        'position': 100.0,
        'berthing_time': 10.0,
        'departure_time': 20.0,
        'uses_shore_power': True
    }

    env_state = {
        'current_allocations': [],
        'berth_length': 2000
    }

    reward = calculator.calculate(vessel, allocation, env_state)
    print(f"   Calculated reward: {reward:.4f}")
    assert isinstance(reward, (int, float))

    print("\n✅ Reward module tests passed!")


def test_integration():
    """Test full system integration."""
    print("\n" + "="*60)
    print("Testing Full System Integration")
    print("="*60)

    from environment import BerthAllocationEnv
    from agents import MATD3Agent, ReplayBuffer

    # Configuration
    config = {
        'environment': {
            'berth_length': 2000,
            'planning_horizon': 7,
            'max_vessels': 5,
            'safe_distance': 20,
            'max_wait_time': 48.0,
            'shore_power': {
                'segments': [
                    {'start': 0, 'end': 1000, 'capacity': 5000},
                    {'start': 1000, 'end': 2000, 'capacity': 5000},
                ],
                'emission_factor_ship': 2500,
                'emission_factor_shore': 800
            },
            'rewards': {f'c{i}': 1.0 for i in range(1, 9)},
            'vessel_generation': {
                'mode': 'simple',
                'peak_hours': [6, 12, 18],
                'peak_rate': 2.0
            },
            'seed': 42
        },
        'agent': {
            'obs_dim': 17,
            'action_dim': 3,
            'num_agents': 5,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'berth_length': 2000,
            'max_wait_time': 48.0,
            'device': 'cpu'
        }
    }

    # Create environment
    print("\n1. Creating environment...")
    env = BerthAllocationEnv(config['environment'])
    print(f"   ✓ Environment created")

    # Create agent
    print("\n2. Creating agent...")
    agent = MATD3Agent(config['agent'])
    print(f"   ✓ Agent created")

    # Create replay buffer
    print("\n3. Creating replay buffer...")
    buffer = ReplayBuffer(
        capacity=10000,
        obs_dim=17,
        action_dim=3,
        num_agents=5
    )
    print(f"   ✓ Buffer created")

    # Run episode
    print("\n4. Running test episode...")
    observations, _ = env.reset()
    episode_reward = 0

    for step in range(10):
        # Select actions
        actions = agent.select_action(observations, explore=True)

        # Step
        next_observations, rewards, done, truncated, infos = env.step(actions)

        # Store in buffer
        if len(observations) > 0:
            full_obs = {i: observations.get(i, np.zeros(17, dtype=np.float32))
                       for i in range(5)}
            full_acts = {i: actions.get(i, np.zeros(3, dtype=np.float32))
                        for i in range(5)}
            full_rews = {i: rewards.get(i, 0.0) for i in range(5)}
            full_next_obs = {i: next_observations.get(i, np.zeros(17, dtype=np.float32))
                            for i in range(5)}

            buffer.add(
                full_obs, full_acts, full_rews, full_next_obs,
                env.get_global_state(), env.get_global_state(), done
            )

        episode_reward += sum(rewards.values()) if rewards else 0
        observations = next_observations

        if done or truncated:
            break

    print(f"   Episode finished: {step+1} steps, reward={episode_reward:.2f}")
    print(f"   Vessels allocated: {len(env.allocations)}")

    # Test update
    if len(buffer) >= 32:
        print("\n5. Testing agent update...")
        stats = agent.update(buffer, 32)
        print(f"   Update successful: {list(stats.keys())}")

    print("\n✅ Full integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "🧪 " * 30)
    print("MATD3 Berth Allocation System - Integration Tests")
    print("🧪 " * 30)

    try:
        test_environment()
        test_agents()
        test_rewards()
        test_integration()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nThe system is ready to use. Try:")
        print("  python main.py --mode train --config config/default_config.yaml")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
