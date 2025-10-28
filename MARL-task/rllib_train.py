"""
RLlib训练脚本 - 多智能体强化学习
支持算法: PPO, SAC (推荐用于连续动作空间)
用于泊位分配多智能体强化学习
"""
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
import os

# 导入环境
from rllib_env.berth_allocation_env import BerthAllocationMultiAgentEnv


def parse_args():
    parser = argparse.ArgumentParser(description='RLlib训练脚本')
    parser.add_argument('--algo', type=str, default='SAC',
                       help='算法名称 (PPO, SAC)')
    parser.add_argument('--num-vessels', type=int, default=10,
                       help='船舶数量')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='训练步数')
    parser.add_argument('--cpus', type=int, default=4,
                       help='CPU核心数')
    parser.add_argument('--gpus', type=int, default=0,
                       help='GPU数量（0=CPU训练）')
    parser.add_argument('--workers', type=int, default=2,
                       help='并行worker数量')
    parser.add_argument('--local', action='store_true',
                       help='本地Mac测试模式（小规模）')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("RLlib泊位分配训练")
    print("=" * 60)
    print(f"算法: {args.algo}")
    print(f"船舶数量: {args.num_vessels}")
    print(f"训练步数: {args.timesteps}")
    print(f"CPUs: {args.cpus}, GPUs: {args.gpus}")
    print("=" * 60)

    # 初始化Ray
    ray.init(
        num_cpus=args.cpus,
        num_gpus=args.gpus,
        ignore_reinit_error=True,
    )

    # 环境配置
    env_config = {
        'berth_length': 2000,
        'max_vessels': args.num_vessels,
        'planning_horizon': 168,
        'shore_power_enabled': True,
        'generation_mode': 'realistic' if not args.local else 'simple',
        'reward_weights': {
            'c1_base': 100.0,
            'c2_waiting': 10.0,
            'c3_emission': 0.01,
            'c4_shore_power': 50.0,
            'c5_utilization': 200.0,
            'c6_spacing': 30.0,
        }
    }

    # 创建临时环境获取空间信息
    temp_env = BerthAllocationMultiAgentEnv(env_config)
    obs, _ = temp_env.reset()
    num_agents = len(obs)

    print(f"\n环境信息:")
    print(f"  智能体数量: {num_agents}")
    print(f"  观测空间: {temp_env.observation_space}")
    print(f"  动作空间: {temp_env.action_space}")

    # 定义策略（简化版：所有智能体共享一个策略）
    policies = {
        "shared_policy": PolicySpec(
            observation_space=temp_env.observation_space,
            action_space=temp_env.action_space,
        )
    }

    # 策略映射函数（所有智能体使用同一策略）
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # 配置算法
    if args.algo.upper() == 'PPO':
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=BerthAllocationMultiAgentEnv,
                env_config=env_config,
            )
            .framework("torch")
            .training(
                lr=5e-5 if not args.local else 1e-3,  # 本地测试用较大学习率
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size_per_learner=512 if not args.local else 256,
                use_gae=True,
                use_kl_loss=True,
                kl_target=0.01,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .resources(
                num_gpus=args.gpus,
            )
            .env_runners(
                num_env_runners=args.workers,
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=1,
            )
            .reporting(
                min_sample_timesteps_per_iteration=1000,
            )
        )
    elif args.algo.upper() == 'SAC':
        # SAC更适合连续动作空间
        config = (
            SACConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=BerthAllocationMultiAgentEnv,
                env_config=env_config,
            )
            .framework("torch")
            .training(
                lr=3e-4 if not args.local else 1e-3,
                gamma=0.99,
                tau=0.005,  # Polyak averaging coefficient
                target_entropy="auto",  # 自动调整熵系数
                initial_alpha=1.0,
                train_batch_size=256 if not args.local else 64,
                target_network_update_freq=1,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": 100000 if not args.local else 10000,
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .resources(
                num_gpus=args.gpus,
            )
            .env_runners(
                num_env_runners=args.workers,
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=1,
            )
            .reporting(
                min_sample_timesteps_per_iteration=1000,
            )
        )
    else:
        raise ValueError(f"不支持的算法: {args.algo}. 支持的算法: PPO, SAC")

    # 训练
    print("\n开始训练...")
    print("-" * 60)

    # 使用绝对路径
    storage_path = os.path.abspath("./ray_results")
    os.makedirs(storage_path, exist_ok=True)

    results = tune.run(
        args.algo.upper(),
        config=config.to_dict(),
        stop={
            "timesteps_total": args.timesteps,
        },
        checkpoint_freq=10 if not args.local else 5,
        checkpoint_at_end=True,
        storage_path=storage_path,
        verbose=1,
        metric="env_runners/episode_return_mean",
        mode="max",
    )

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"结果目录: {storage_path}")

    # 获取最佳结果
    best_trial = results.get_best_trial("env_runners/episode_return_mean", mode="max")
    if best_trial:
        print(f"最佳Trial: {best_trial.trial_id}")
        print(f"最佳奖励: {best_trial.last_result['env_runners/episode_return_mean']:.2f}")
        if results.best_checkpoint:
            print(f"最佳检查点: {results.best_checkpoint}")

    # 关闭Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
