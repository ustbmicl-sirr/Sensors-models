"""Task management service."""

from typing import Dict, Optional
import sys
import os

# Add parent directory to path to import from main project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from environment import BerthAllocationEnv, Vessel


class TaskManager:
    """Manages task creation and storage."""

    def __init__(self):
        self.tasks: Dict[str, dict] = {}

    def create_task(self, task_id: str, config: dict) -> dict:
        """
        Create a new berth allocation task.

        Args:
            task_id: Unique task identifier
            config: Task configuration dictionary

        Returns:
            Task data with vessels and environment info
        """
        # Build environment configuration
        env_config = {
            'berth_length': config.get('berth_length', 2000),
            'planning_horizon': config['planning_horizon'],
            'max_vessels': config['num_vessels'],
            'safe_distance': 20,
            'max_wait_time': 48.0,
            'target_utilization': 0.7,
            'shore_power': self._build_shore_power_config(config),
            'vessel_generation': {
                'mode': config.get('generation_mode', 'realistic'),
                'seed': config.get('seed', 42),
                'peak_hours': [6, 12, 18],
                'peak_rate': 2.0,
                'size_distribution': [0.3, 0.5, 0.2],
                'shore_power_ratio': 0.6
            },
            'rewards': {
                'c1': 10.0,
                'c2': 5.0,
                'c3': 8.0,
                'c4': 6.0,
                'c5': 4.0,
                'c6': 3.0,
                'c7': 0.0,
                'c8': 20.0
            },
            'seed': config.get('seed', 42)
        }

        # Create environment and generate vessels
        env = BerthAllocationEnv(env_config)
        observations, info = env.reset()

        # Serialize vessel data
        vessels_data = [
            {
                'id': v.id,
                'length': float(v.length),
                'draft': float(v.draft),
                'arrival_time': float(v.arrival_time),
                'operation_time': float(v.operation_time),
                'priority': int(v.priority),
                'can_use_shore_power': bool(v.can_use_shore_power),
                'power_requirement': float(v.power_requirement)
            }
            for v in env.vessels
        ]

        # Serialize environment data
        environment_data = {
            'berth_length': float(env.berth_length),
            'shore_power_segments': [
                {
                    'id': seg.id,
                    'start': float(seg.start),
                    'end': float(seg.end),
                    'capacity': float(seg.capacity)
                }
                for seg in env.shore_power.segments
            ]
        }

        # Store task
        task_data = {
            'config': env_config,
            'vessels': vessels_data,
            'environment': environment_data
        }

        self.tasks[task_id] = task_data
        return task_data

    def get_task(self, task_id: str) -> Optional[dict]:
        """
        Retrieve a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task data or None if not found
        """
        return self.tasks.get(task_id)

    def _build_shore_power_config(self, config: dict) -> dict:
        """Build shore power configuration."""
        if not config.get('shore_power_enabled', True):
            return {
                'segments': [],
                'emission_factor_ship': 2500,
                'emission_factor_shore': 800
            }

        # Default shore power configuration: 5 segments of 400m each
        berth_length = config.get('berth_length', 2000)
        segment_length = berth_length / 5

        return {
            'segments': [
                {
                    'start': i * segment_length,
                    'end': (i + 1) * segment_length,
                    'capacity': 5000
                }
                for i in range(5)
            ],
            'emission_factor_ship': 2500,
            'emission_factor_shore': 800
        }


# Global instance
task_manager = TaskManager()
