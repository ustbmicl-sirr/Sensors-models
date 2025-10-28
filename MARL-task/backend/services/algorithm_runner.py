"""Algorithm execution service."""

import os
import sys
import numpy as np
import time
from typing import Dict, AsyncIterator

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from environment import BerthAllocationEnv, Vessel
from agents import MATD3Agent
from rewards import RewardCalculator
from backend.utils import algorithm_logger


class AlgorithmRunner:
    """Executes allocation algorithms."""

    def __init__(self):
        self.agents_cache: Dict[str, MATD3Agent] = {}
        self.logger = algorithm_logger

    def run(self, task_data: dict, algorithm: str, model_path: str = None) -> dict:
        """
        Run allocation algorithm synchronously.

        Args:
            task_data: Task data from TaskManager
            algorithm: Algorithm name ('MATD3', 'Greedy', 'FCFS')
            model_path: Path to trained model (for MATD3)

        Returns:
            Dictionary with allocations and metrics
        """
        start_time = time.time()
        task_id = task_data.get('task_id', 'unknown')

        self.logger.info(
            f"Starting {algorithm} algorithm",
            task_id=task_id,
            algorithm=algorithm,
            num_vessels=len(task_data['vessels'])
        )

        env_config = task_data['config']

        # Rebuild environment with original vessels
        env = BerthAllocationEnv(env_config)
        vessels = [Vessel(**v) for v in task_data['vessels']]
        env.reset(options={'vessels': vessels})

        # Run selected algorithm
        try:
            if algorithm == 'MATD3':
                self._run_matd3(env, model_path)
            elif algorithm == 'Greedy':
                self._run_greedy(env)
            elif algorithm == 'FCFS':
                self._run_fcfs(env)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            self.logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                task_id=task_id,
                algorithm=algorithm
            )
            raise

        # Extract allocations
        allocations = self._extract_allocations(env)

        # Calculate metrics
        calculator = RewardCalculator(env_config)
        metrics = calculator.calculate_metrics(env.allocations)

        duration_ms = (time.time() - start_time) * 1000

        self.logger.info(
            f"{algorithm} completed successfully",
            task_id=task_id,
            algorithm=algorithm,
            duration_ms=duration_ms,
            vessels_allocated=len(allocations),
            berth_utilization=metrics['berth_utilization']
        )

        return {
            'allocations': allocations,
            'metrics': metrics
        }

    async def run_streaming(
        self,
        task_data: dict,
        algorithm: str,
        model_path: str = None
    ) -> AsyncIterator[dict]:
        """
        Run algorithm with streaming updates via WebSocket.

        Args:
            task_data: Task data
            algorithm: Algorithm name
            model_path: Model path for MATD3

        Yields:
            Dictionary messages for WebSocket
        """
        if algorithm != 'MATD3':
            raise ValueError("Only MATD3 supports streaming mode")

        env_config = task_data['config']
        env = BerthAllocationEnv(env_config)
        vessels = [Vessel(**v) for v in task_data['vessels']]
        env.reset(options={'vessels': vessels})

        # Load agent with actual number of vessels
        num_agents = len(env.vessels)
        agent = self._load_agent(model_path, env_config, num_agents)

        # Execute step by step
        observations, _ = env.reset()
        step = 0

        while observations:
            # Select actions
            actions = agent.select_action(observations, explore=False)

            # Execute step
            next_obs, rewards, done, truncated, infos = env.step(actions)

            # Send allocation events
            for vessel_id, info in infos.items():
                if isinstance(info, dict) and info.get('allocated'):
                    yield {
                        'type': 'allocation',
                        'step': step,
                        'vessel_id': vessel_id,
                        'position': float(info['position']),
                        'berthing_time': float(info['berthing_time']),
                        'uses_shore_power': bool(info.get('uses_shore_power', False))
                    }

            # Send metrics update every 5 steps
            if step % 5 == 0 and env.allocations:
                calculator = RewardCalculator(env_config)
                metrics = calculator.calculate_metrics(env.allocations)

                yield {
                    'type': 'metrics_update',
                    'step': step,
                    'metrics': {
                        'berth_utilization': float(metrics['berth_utilization']),
                        'avg_waiting_time': float(metrics['avg_waiting_time']),
                        'total_emissions': float(metrics['total_emissions']),
                        'shore_power_usage_rate': float(metrics['shore_power_usage_rate']),
                        'num_vessels': int(metrics['num_vessels'])
                    }
                }

            observations = next_obs
            step += 1

            if done or truncated:
                break

        # Send final metrics
        if env.allocations:
            calculator = RewardCalculator(env_config)
            metrics = calculator.calculate_metrics(env.allocations)

            yield {
                'type': 'metrics_update',
                'step': step,
                'metrics': {
                    'berth_utilization': float(metrics['berth_utilization']),
                    'avg_waiting_time': float(metrics['avg_waiting_time']),
                    'total_emissions': float(metrics['total_emissions']),
                    'shore_power_usage_rate': float(metrics['shore_power_usage_rate']),
                    'num_vessels': int(metrics['num_vessels'])
                }
            }

    def _run_matd3(self, env: BerthAllocationEnv, model_path: str = None):
        """Run MATD3 algorithm."""
        # Use actual number of vessels instead of max_vessels
        num_agents = len(env.vessels)
        agent = self._load_agent(model_path, env.config, num_agents)

        observations, _ = env.reset()

        while observations:
            actions = agent.select_action(observations, explore=False)
            next_obs, _, done, truncated, _ = env.step(actions)
            observations = next_obs

            if done or truncated:
                break

    def _run_greedy(self, env: BerthAllocationEnv):
        """
        Run greedy baseline algorithm.

        Strategy: Assign vessels in arrival order to earliest available position.
        """
        for vessel in sorted(env.vessels, key=lambda v: v.arrival_time):
            # Find earliest available position
            position = self._find_earliest_position(vessel, env.allocations, env.berth_length)

            # Create allocation
            allocation = {
                'vessel': vessel,
                'vessel_id': vessel.id,
                'position': position,
                'berthing_time': vessel.arrival_time,
                'departure_time': vessel.arrival_time + vessel.operation_time,
                'uses_shore_power': False,
                'shore_power_segments': []
            }

            env.allocations.append(allocation)

    def _run_fcfs(self, env: BerthAllocationEnv):
        """
        Run First-Come-First-Served baseline algorithm.

        Strategy: Assign vessels linearly along berth in arrival order.
        """
        current_position = 0.0

        for vessel in sorted(env.vessels, key=lambda v: v.arrival_time):
            # Simple linear allocation
            allocation = {
                'vessel': vessel,
                'vessel_id': vessel.id,
                'position': current_position,
                'berthing_time': vessel.arrival_time,
                'departure_time': vessel.arrival_time + vessel.operation_time,
                'uses_shore_power': False,
                'shore_power_segments': []
            }

            env.allocations.append(allocation)

            # Move to next position
            current_position += vessel.length + 20  # 20m safety distance

            # Wrap around if exceed berth length
            if current_position > env.berth_length - 100:
                current_position = 0.0

    def _load_agent(self, model_path: str, env_config: dict, num_agents: int = None) -> MATD3Agent:
        """Load or create MATD3 agent."""
        if model_path and model_path in self.agents_cache:
            return self.agents_cache[model_path]

        # Use provided num_agents or fall back to max_vessels
        if num_agents is None:
            num_agents = env_config['max_vessels']

        # Create agent configuration
        agent_config = {
            'obs_dim': 17,
            'action_dim': 3,
            'num_agents': num_agents,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'grad_clip': 1.0,
            'berth_length': env_config['berth_length'],
            'max_wait_time': env_config.get('max_wait_time', 48.0),
            'exploration_noise': {
                'position': 0.1,
                'time': 0.1,
                'probability': 0.1
            },
            'device': 'cpu'
        }

        agent = MATD3Agent(agent_config)

        # Load model if path provided and exists
        if model_path and os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using randomly initialized agent")

        if model_path:
            self.agents_cache[model_path] = agent

        return agent

    def _extract_allocations(self, env: BerthAllocationEnv) -> list:
        """Extract allocation results from environment."""
        return [
            {
                'vessel_id': int(alloc['vessel'].id),
                'position': float(alloc['position']),
                'berthing_time': float(alloc['berthing_time']),
                'departure_time': float(alloc['departure_time']),
                'uses_shore_power': bool(alloc.get('uses_shore_power', False)),
                'waiting_time': float(alloc['berthing_time'] - alloc['vessel'].arrival_time)
            }
            for alloc in env.allocations
        ]

    def _find_earliest_position(
        self,
        vessel: Vessel,
        allocations: list,
        berth_length: float
    ) -> float:
        """
        Find earliest available position for vessel.

        Args:
            vessel: Vessel to allocate
            allocations: Current allocations
            berth_length: Total berth length

        Returns:
            Available position in meters
        """
        # Try positions in 50m increments
        for pos in np.arange(0, berth_length - vessel.length, 50):
            valid = True

            # Check for overlaps with existing allocations
            for alloc in allocations:
                if self._has_overlap(
                    pos,
                    vessel.length,
                    alloc['position'],
                    alloc['vessel'].length
                ):
                    valid = False
                    break

            if valid:
                return float(pos)

        # If no position found, return 0 (will likely cause overlap)
        return 0.0

    def _has_overlap(
        self,
        pos1: float,
        len1: float,
        pos2: float,
        len2: float,
        safe_distance: float = 20.0
    ) -> bool:
        """Check if two vessel positions overlap (including safe distance)."""
        return not (pos1 + len1 + safe_distance <= pos2 or
                   pos2 + len2 + safe_distance <= pos1)


# Global instance
algorithm_runner = AlgorithmRunner()
