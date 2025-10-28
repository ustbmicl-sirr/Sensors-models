"""
Evaluator for trained MATD3 agent.
"""

import numpy as np
import time
from typing import Dict
from rewards import RewardCalculator


class Evaluator:
    """
    Evaluator for MATD3 agent on berth allocation.
    """

    def __init__(self, env, agent, config: dict):
        """
        Initialize evaluator.

        Args:
            env: Environment
            agent: MATD3 agent
            config: Configuration dictionary
        """
        self.env = env
        self.agent = agent
        self.config = config

        # Evaluation config
        self.num_episodes = config.get('evaluation', {}).get('num_episodes', 10)

        # Reward calculator for metrics
        self.reward_calculator = RewardCalculator(config['environment'])

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            num_episodes: Number of episodes (overrides config)

        Returns:
            Dictionary of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        # Set agent to eval mode
        self.agent.set_eval_mode()

        all_metrics = []
        all_inference_times = []

        for episode in range(num_episodes):
            # Reset environment
            observations, _ = self.env.reset()

            episode_allocations = []
            inference_times = []

            # Run episode
            done = False
            while not done:
                # Measure inference time
                start_time = time.time()
                actions = self.agent.select_action(observations, explore=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Step
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)

                # Update
                observations = next_observations
                done = terminated or truncated

            # Collect allocations
            episode_allocations = self.env.allocations

            # Calculate metrics
            metrics = self._calculate_metrics(episode_allocations)
            metrics['avg_inference_time'] = np.mean(inference_times) if inference_times else 0

            all_metrics.append(metrics)
            all_inference_times.extend(inference_times)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        # Set agent back to train mode
        self.agent.set_train_mode()

        return aggregated_metrics

    def _calculate_metrics(self, allocations) -> Dict[str, float]:
        """
        Calculate performance metrics for allocations.

        Args:
            allocations: List of allocation dictionaries

        Returns:
            Dictionary of metrics
        """
        if not allocations:
            return {
                'berth_utilization': 0.0,
                'avg_waiting_time': 0.0,
                'total_emissions': 0.0,
                'shore_power_usage_rate': 0.0,
                'num_vessels': 0
            }

        # Calculate waiting times
        waiting_times = []
        waiting_by_priority = {1: [], 2: [], 3: [], 4: []}

        for alloc in allocations:
            vessel = alloc['vessel']
            wait_time = alloc['berthing_time'] - vessel.arrival_time
            waiting_times.append(wait_time)

            if vessel.priority in waiting_by_priority:
                waiting_by_priority[vessel.priority].append(wait_time)

        # Berth utilization (space-time)
        total_berth_time = sum(
            (alloc['departure_time'] - alloc['berthing_time']) * alloc['vessel'].length
            for alloc in allocations
        )
        max_berth_time = self.env.berth_length * self.env.planning_horizon_hours
        berth_utilization = total_berth_time / max_berth_time

        # Emissions
        total_emissions = sum(
            self.reward_calculator._calculate_emissions(
                alloc['vessel'],
                alloc['uses_shore_power']
            )
            for alloc in allocations
        )

        # Shore power usage
        shore_power_count = sum(1 for alloc in allocations if alloc['uses_shore_power'])
        shore_power_capable = sum(
            1 for alloc in allocations if alloc['vessel'].can_use_shore_power
        )
        shore_power_usage_rate = (
            shore_power_count / shore_power_capable if shore_power_capable > 0 else 0.0
        )

        metrics = {
            'berth_utilization': berth_utilization,
            'avg_waiting_time': np.mean(waiting_times),
            'waiting_time_std': np.std(waiting_times),
            'total_emissions': total_emissions,
            'shore_power_usage_rate': shore_power_usage_rate,
            'num_vessels': len(allocations)
        }

        # Add priority-specific waiting times
        for priority in [1, 2, 3, 4]:
            if waiting_by_priority[priority]:
                metrics[f'waiting_time_p{priority}'] = np.mean(waiting_by_priority[priority])
            else:
                metrics[f'waiting_time_p{priority}'] = 0.0

        return metrics

    def _aggregate_metrics(self, metrics_list) -> Dict[str, float]:
        """
        Aggregate metrics across episodes.

        Args:
            metrics_list: List of metric dictionaries

        Returns:
            Aggregated metrics with mean and std
        """
        aggregated = {}

        # Get all metric keys
        keys = set()
        for metrics in metrics_list:
            keys.update(metrics.keys())

        # Aggregate each metric
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)

        return aggregated

    def evaluate_and_save(self, save_path: str = None):
        """
        Evaluate and save detailed results.

        Args:
            save_path: Path to save results
        """
        # Evaluate
        metrics = self.evaluate()

        # Save if path provided
        if save_path:
            np.savez(
                save_path,
                **metrics
            )

        return metrics
