#!/usr/bin/env python3
"""
Main entry point for berth allocation with MATD3.

Usage:
    python main.py --mode train --config config/default_config.yaml
    python main.py --mode eval --model results/models/model_1000.pth
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from environment import BerthAllocationEnv
from agents import MATD3Agent, ReplayBuffer
from training import Trainer, Evaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: dict, **overrides) -> dict:
    """Merge configuration with overrides."""
    config = base_config.copy()

    # Flatten environment config for agent
    agent_config = config['agent'].copy()
    agent_config.update(config['environment'])
    config['agent'] = agent_config

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


def train(config: dict):
    """Train MATD3 agent."""
    print("="*60)
    print("Training MATD3 for Berth Allocation")
    print("="*60)

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env_config = config['environment'].copy()
    env_config['seed'] = seed
    env = BerthAllocationEnv(env_config)

    print(f"\nEnvironment:")
    print(f"  Berth length: {env.berth_length}m")
    print(f"  Planning horizon: {env.planning_horizon_days} days")
    print(f"  Max vessels: {env.max_vessels}")
    print(f"  Shore power segments: {len(env.shore_power.segments)}")

    # Create agent
    agent_config = config['agent'].copy()
    agent_config['device'] = config.get('device', 'cpu')
    agent = MATD3Agent(agent_config)

    print(f"\nAgent:")
    print(f"  Observation dim: {agent.obs_dim}")
    print(f"  Action dim: {agent.action_dim}")
    print(f"  Num agents: {agent.num_agents}")
    print(f"  Device: {agent.device}")

    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=config['training']['buffer_size'],
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        num_agents=agent.num_agents,
        device=agent.device
    )

    print(f"\nReplay buffer size: {config['training']['buffer_size']}")

    # Create trainer
    trainer = Trainer(env, agent, buffer, config)

    # Train
    print(f"\nStarting training for {config['training']['num_episodes']} episodes...")
    trainer.train()

    print("\n✓ Training complete!")


def evaluate(config: dict, model_path: str):
    """Evaluate trained agent."""
    print("="*60)
    print("Evaluating MATD3 Agent")
    print("="*60)

    # Load model
    print(f"\nLoading model from: {model_path}")

    # Create environment
    env_config = config['environment'].copy()
    env = BerthAllocationEnv(env_config)

    # Create agent
    agent_config = config['agent'].copy()
    agent_config['device'] = config.get('device', 'cpu')
    agent = MATD3Agent(agent_config)

    # Load weights
    agent.load(model_path)
    agent.set_eval_mode()

    # Create evaluator
    evaluator = Evaluator(env, agent, config)

    # Evaluate
    print(f"\nEvaluating for {config['evaluation']['num_episodes']} episodes...")
    metrics = evaluator.evaluate()

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Berth Allocation with MATD3')

    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'eval'],
                       help='Mode: train or eval')

    parser.add_argument('--config', type=str,
                       default='config/default_config.yaml',
                       help='Path to configuration file')

    parser.add_argument('--model', type=str,
                       help='Path to model checkpoint (for eval mode)')

    parser.add_argument('--device', type=str,
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    parser.add_argument('--seed', type=int,
                       help='Random seed')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed

    # Merge configs
    config = merge_configs(config)

    # Create directories
    Path(config['training']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['figure_dir']).mkdir(parents=True, exist_ok=True)

    # Run
    if args.mode == 'train':
        train(config)
    elif args.mode == 'eval':
        if not args.model:
            raise ValueError("--model is required for eval mode")
        evaluate(config, args.model)


if __name__ == '__main__':
    main()
