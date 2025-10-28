#!/usr/bin/env python3
"""
RLlib高级训练脚本 - 并行优化版本
支持分布式训练、GPU加速、性能监控
"""

import argparse
import os
import sys
import time
import multiprocessing
from datetime import datetime
from typing import Dict, Optional

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rllib_env.berth_allocation_env import BerthAllocationMultiAgentEnv


def register_berth_env():
    """注册自定义环境"""
    def env_creator(env_config):
        return BerthAllocationMultiAgentEnv(env_config)

    register_env("berth_allocation", env_creator)


def get_optimal_resources():
    """
    自动检测并返回最优资源配置

    Returns:
        dict: 资源配置
    """
    num_cpus = multiprocessing.cpu_count()

    try:
        import torch
        num_gpus = torch.cuda.device_count()
        gpu_available = torch.cuda.is_available()
    except:
        num_gpus = 0
        gpu_available = False

    # 计算最优worker数量
    if num_gpus > 0:
        # 有GPU: worker专注采样,learner用GPU
        num_workers = min(num_cpus - 2, num_gpus * 8)
    else:
        # 无GPU: 减少worker避免竞争
        num_workers = max(2, num_cpus // 2)

    return {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "gpu_available": gpu_available,
        "recommended_workers": num_workers,
    }


def create_training_config(
    algo_name: str,
    env_config: Dict,
    num_gpus: int = 0,
    num_workers: int = 2,
    distributed: bool = False,
    optimize_for: str = "speed"  # "speed" or "quality"
) -> object:
    """
    创建训练配置

    Args:
        algo_name: 算法名称
        env_config: 环境配置
        num_gpus: GPU数量
        num_workers: Worker数量
        distributed: 是否分布式训练
        optimize_for: 优化目标 ("speed"或"quality")

    Returns:
        配置对象
    """

    # 选择算法
    if algo_name.upper() == "SAC":
        config = SACConfig()
    elif algo_name.upper() == "PPO":
        config = PPOConfig()
    elif algo_name.upper() == "APPO":
        config = APPOConfig()
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    # 环境配置
    config.environment(
        env="berth_allocation",
        env_config=env_config,
    )

    # 框架配置
    config.framework("torch")

    # 资源配置
    if distributed:
        # 分布式配置
        config.resources(
            num_gpus=num_gpus,
            num_cpus_for_driver=min(8, multiprocessing.cpu_count() // 4),
        )
    else:
        # 单机配置
        config.resources(
            num_gpus=num_gpus,
            num_cpus_for_driver=2,
        )

    # Rollout配置 - 根据优化目标调整
    if optimize_for == "speed":
        # 速度优先: 更多worker,更长fragment
        config.rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=2,
            rollout_fragment_length=1000,
            compress_observations=True,  # 压缩观测节省带宽
        )
    else:
        # 质量优先: 更大batch,更多SGD迭代
        config.rollouts(
            num_rollout_workers=max(4, num_workers // 2),
            num_envs_per_worker=1,
            rollout_fragment_length=500,
        )

    # 训练配置
    num_envs = config.num_rollout_workers * config.num_envs_per_worker
    fragment_length = config.rollout_fragment_length

    if algo_name.upper() == "SAC":
        # SAC特定配置
        if optimize_for == "speed":
            train_batch_size = min(8000, num_envs * fragment_length)
        else:
            train_batch_size = min(16000, num_envs * fragment_length * 2)

        config.training(
            train_batch_size=train_batch_size,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000 if optimize_for == "speed" else 1000000,
            },
            optimization={
                "actor_learning_rate": 3e-4,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 3e-4,
            },
            tau=0.005,
            target_network_update_freq=1,
            num_steps_sampled_before_learning_starts=1000,
        )

    elif algo_name.upper() == "PPO":
        # PPO特定配置
        train_batch_size = num_envs * fragment_length

        if optimize_for == "speed":
            num_sgd_iter = 5
            sgd_minibatch_size = 128
        else:
            num_sgd_iter = 10
            sgd_minibatch_size = 256

        config.training(
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )

    elif algo_name.upper() == "APPO":
        # APPO异步训练配置
        config.training(
            train_batch_size=500,
            num_sgd_iter=1,
            lr=3e-4,
            vtrace=True,
        )

    # 模型配置
    config.model = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    # 评估配置
    config.evaluation(
        evaluation_interval=10,
        evaluation_duration=10,
        evaluation_num_workers=min(2, num_workers // 4) if num_workers > 4 else 0,
        evaluation_config={
            "explore": False,
        },
    )

    # 调试配置
    config.debugging(
        log_level="INFO",
    )

    # 报告配置
    config.reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000,
    )

    return config


def train_with_monitoring(
    algo,
    num_iterations: int,
    checkpoint_freq: int = 50,
    results_dir: str = "./ray_results",
):
    """
    带监控的训练循环

    Args:
        algo: RLlib算法实例
        num_iterations: 训练迭代次数
        checkpoint_freq: 检查点保存频率
        results_dir: 结果保存目录
    """

    best_reward = -float('inf')
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for iteration in range(num_iterations):
        iter_start = time.time()

        # 训练一次迭代
        result = algo.train()

        iter_time = time.time() - iter_start
        total_time = time.time() - start_time

        # 提取关键指标
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)
        timesteps_total = result.get("timesteps_total", 0)

        # 性能指标
        info = result.get("info", {})
        learner_info = info.get("learner", {})

        # 打印进度
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"  Episode Reward Mean: {episode_reward_mean:.2f}")
        print(f"  Episode Length Mean: {episode_len_mean:.2f}")
        print(f"  Timesteps Total: {timesteps_total}")
        print(f"  Iteration Time: {iter_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s ({total_time/3600:.2f}h)")

        # 采样性能
        sampler_results = result.get("sampler_results", {})
        if sampler_results:
            episode_reward = sampler_results.get("episode_reward_mean", 0)
            print(f"  Sampler Reward: {episode_reward:.2f}")

        # 计算FPS
        if iter_time > 0:
            samples_per_iter = result.get("num_env_steps_sampled", 0)
            fps = samples_per_iter / iter_time
            print(f"  Sampling FPS: {fps:.0f}")

        # 保存最佳模型
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_path = algo.save(results_dir)
            print(f"  💾 New best! Saved to: {checkpoint_path}")

        # 定期保存检查点
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(results_dir)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        print()

    # 训练完成
    total_hours = (time.time() - start_time) / 3600
    print(f"\n{'='*70}")
    print(f"训练完成!")
    print(f"  总时间: {total_hours:.2f}小时")
    print(f"  最佳奖励: {best_reward:.2f}")
    print(f"  总采样步数: {timesteps_total}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RLlib高级训练脚本 - 支持分布式和并行优化"
    )

    # 算法参数
    parser.add_argument("--algo", type=str, default="SAC",
                       choices=["SAC", "PPO", "APPO"],
                       help="强化学习算法")

    # 环境参数
    parser.add_argument("--num-vessels", type=int, default=10,
                       help="船舶数量")
    parser.add_argument("--planning-horizon", type=int, default=7,
                       help="规划周期(天)")
    parser.add_argument("--berth-length", type=float, default=2000.0,
                       help="泊位长度(米)")

    # 训练参数
    parser.add_argument("--iterations", type=int, default=100,
                       help="训练迭代次数")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="检查点保存频率")

    # 资源参数
    parser.add_argument("--gpus", type=int, default=None,
                       help="GPU数量 (None=自动检测)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Worker数量 (None=自动计算)")
    parser.add_argument("--auto-resources", action="store_true",
                       help="自动检测并使用最优资源配置")

    # 优化参数
    parser.add_argument("--optimize-for", type=str, default="speed",
                       choices=["speed", "quality"],
                       help="优化目标: speed(速度) 或 quality(质量)")

    # 分布式参数
    parser.add_argument("--distributed", action="store_true",
                       help="启用分布式训练 (需要Ray集群)")
    parser.add_argument("--ray-address", type=str, default=None,
                       help="Ray集群地址 (默认: 本地)")

    # 调试参数
    parser.add_argument("--local", action="store_true",
                       help="本地调试模式 (单线程)")
    parser.add_argument("--profile", action="store_true",
                       help="性能分析模式")

    # 输出参数
    parser.add_argument("--results-dir", type=str, default="./ray_results",
                       help="结果保存目录")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")

    args = parser.parse_args()

    # 显示资源信息
    resources = get_optimal_resources()
    print(f"\n{'='*70}")
    print(f"系统资源检测")
    print(f"{'='*70}")
    print(f"  CPU核数: {resources['num_cpus']}")
    print(f"  GPU数量: {resources['num_gpus']}")
    print(f"  GPU可用: {resources['gpu_available']}")
    print(f"  推荐Worker数: {resources['recommended_workers']}")
    print(f"{'='*70}\n")

    # 资源配置
    if args.auto_resources:
        num_gpus = resources['num_gpus'] if resources['gpu_available'] else 0
        num_workers = resources['recommended_workers']
        print(f"🤖 自动资源配置:")
        print(f"  使用 {num_gpus} 个GPU")
        print(f"  使用 {num_workers} 个Worker\n")
    else:
        num_gpus = args.gpus if args.gpus is not None else 0
        num_workers = args.workers if args.workers is not None else 2

    # 初始化Ray
    if args.local:
        ray.init(local_mode=True, ignore_reinit_error=True)
        print("📍 本地调试模式\n")
    elif args.distributed and args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        print(f"🌐 已连接到Ray集群: {args.ray_address}\n")
    else:
        ray.init(
            num_cpus=resources['num_cpus'],
            num_gpus=resources['num_gpus'],
            ignore_reinit_error=True
        )
        print("🖥️  单机模式\n")

    # 注册环境
    register_berth_env()

    # 环境配置
    env_config = {
        "num_vessels": args.num_vessels,
        "planning_horizon_days": args.planning_horizon,
        "berth_length": args.berth_length,
    }

    # 创建训练配置
    print(f"{'='*70}")
    print(f"训练配置")
    print(f"{'='*70}")
    print(f"  算法: {args.algo}")
    print(f"  船舶数量: {args.num_vessels}")
    print(f"  规划周期: {args.planning_horizon}天")
    print(f"  泊位长度: {args.berth_length}米")
    print(f"  训练迭代: {args.iterations}")
    print(f"  GPU数量: {num_gpus}")
    print(f"  Worker数量: {num_workers}")
    print(f"  优化目标: {args.optimize_for}")
    print(f"  分布式: {args.distributed}")
    print(f"  结果目录: {args.results_dir}")
    print(f"{'='*70}\n")

    config = create_training_config(
        algo_name=args.algo,
        env_config=env_config,
        num_gpus=num_gpus,
        num_workers=num_workers,
        distributed=args.distributed,
        optimize_for=args.optimize_for,
    )

    # 创建算法实例
    algo = config.build()

    # 性能分析模式
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # 训练
        train_with_monitoring(
            algo=algo,
            num_iterations=args.iterations,
            checkpoint_freq=args.checkpoint_freq,
            results_dir=args.results_dir,
        )

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 性能分析
        if args.profile:
            profiler.disable()
            profiler.print_stats(sort='cumtime')

        # 清理
        algo.stop()
        ray.shutdown()

        print("\n✅ 训练结束，资源已释放\n")


if __name__ == "__main__":
    main()
