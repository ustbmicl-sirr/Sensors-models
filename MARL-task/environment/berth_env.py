"""
Berth allocation environment with continuous berth and shore power.

Multi-agent POMDP environment for berth allocation and shore power assignment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy

from .vessel import Vessel, VesselGenerator
from .shore_power import ShorePowerManager


class BerthAllocationEnv(gym.Env):
    """
    Continuous berth allocation environment with shore power coordination.

    This environment implements a POMDP for multi-agent berth allocation where:
    - State: Partially observable (each vessel sees local features)
    - Action: Continuous (position, waiting time, shore power probability)
    - Reward: Multi-objective (utilization, waiting time, emissions)
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, config: dict):
        """
        Initialize berth allocation environment.

        Args:
            config: Configuration dictionary containing:
                - berth_length: Total berth length (m)
                - planning_horizon: Planning horizon (days)
                - max_vessels: Maximum number of vessels
                - safe_distance: Minimum safe distance between vessels (m)
                - shore_power: Shore power configuration
                - rewards: Reward weights
                - vessel_generation: Vessel generation config
        """
        super().__init__()

        self.config = config
        self.berth_length = config['berth_length']
        self.planning_horizon_days = config['planning_horizon']
        self.planning_horizon_hours = self.planning_horizon_days * 24
        self.max_vessels = config['max_vessels']
        self.safe_distance = config.get('safe_distance', 20)

        # Shore power manager
        self.shore_power = ShorePowerManager(config['shore_power'])

        # Vessel generator
        vessel_gen_config = config.get('vessel_generation', {})
        self.vessel_generator = VesselGenerator(
            vessel_gen_config,
            seed=config.get('seed')
        )

        # Current state
        self.vessels: List[Vessel] = []
        self.allocations: List[Dict] = []  # Completed allocations
        self.current_time = 0.0
        self.pending_vessels: List[int] = []  # Vessel IDs not yet allocated

        # Define spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = config.get('max_episode_steps', 1000)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: 17-dimensional local features
        # [ship_length, arrival_time, priority, can_shore_power,
        #  current_time, waiting_time, operation_time,
        #  shore_power_seg1, seg2, seg3, seg4, seg5, total_shore,
        #  left_neighbor_dist, right_neighbor_dist, available_space_left, available_space_right]

        obs_low = np.array([
            0.0,    # ship_length (normalized)
            0.0,    # arrival_time (normalized)
            0.0,    # priority (normalized)
            0.0,    # can_shore_power
            0.0,    # current_time (normalized)
            0.0,    # waiting_time (normalized)
            0.0,    # operation_time (normalized)
            0.0, 0.0, 0.0, 0.0, 0.0,  # shore power segments usage
            0.0,    # total shore power usage
            0.0,    # left_neighbor_distance (normalized)
            0.0,    # right_neighbor_distance (normalized)
            0.0,    # available_space_left (normalized)
            0.0,    # available_space_right (normalized)
        ], dtype=np.float32)

        obs_high = np.array([
            1.0,    # ship_length
            1.0,    # arrival_time
            1.0,    # priority
            1.0,    # can_shore_power
            1.0,    # current_time
            1.0,    # waiting_time
            1.0,    # operation_time
            1.0, 1.0, 1.0, 1.0, 1.0,  # shore power segments
            1.0,    # total shore power
            1.0,    # left_neighbor
            1.0,    # right_neighbor
            1.0,    # available_left
            1.0,    # available_right
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: [position, waiting_time, shore_power_prob]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (e.g., 'vessels' for custom vessel list)

        Returns:
            observations: Dictionary of observations for each agent
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset state
        self.current_time = 0.0
        self.episode_step = 0
        self.allocations = []
        self.shore_power.reset()

        # Generate or load vessels
        if options and 'vessels' in options:
            self.vessels = options['vessels']
        else:
            generation_mode = self.config.get('vessel_generation', {}).get('mode', 'realistic')
            if generation_mode == 'realistic':
                self.vessels = self.vessel_generator.generate_realistic(
                    self.max_vessels,
                    self.planning_horizon_days
                )
            else:
                self.vessels = self.vessel_generator.generate_simple(
                    self.max_vessels,
                    self.planning_horizon_days
                )

        # Sort by arrival time
        self.vessels.sort(key=lambda v: v.arrival_time)

        # Initialize pending vessels (all vessels initially)
        self.pending_vessels = list(range(len(self.vessels)))

        # Get initial observations
        observations = self._get_observations()

        info = {
            'num_vessels': len(self.vessels),
            'total_length': sum(v.length for v in self.vessels),
            'planning_horizon': self.planning_horizon_hours
        }

        return observations, info

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary mapping vessel_id to action [position, wait_time, shore_power_prob]

        Returns:
            observations: Next observations
            rewards: Rewards for each agent
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            infos: Additional information
        """
        self.episode_step += 1

        rewards = {}
        infos = {}
        newly_allocated = []

        # Process each action
        for vessel_id, action in actions.items():
            if vessel_id not in self.pending_vessels:
                continue  # Already allocated

            vessel = self.vessels[vessel_id]

            # Denormalize action
            position = action[0] * self.berth_length
            wait_time = action[1] * 48.0  # Max 48 hours wait
            shore_power_prob = action[2]

            # Decide shore power usage (stochastic based on probability)
            use_shore_power = (shore_power_prob > 0.5) and vessel.can_use_shore_power

            # Calculate berthing time
            berthing_time = vessel.arrival_time + wait_time
            departure_time = berthing_time + vessel.operation_time

            # Check constraints
            is_valid, violation_reason = self._check_constraints(
                vessel, position, berthing_time, departure_time, use_shore_power
            )

            if is_valid:
                # Allocate berth
                allocation = self._allocate_berth(
                    vessel, position, berthing_time, departure_time, use_shore_power
                )

                # Calculate reward
                reward = self._calculate_reward(vessel, allocation)

                # Update vessel
                vessel.position = position
                vessel.berthing_time = berthing_time
                vessel.departure_time = departure_time
                vessel.waiting_time = wait_time
                vessel.uses_shore_power = use_shore_power

                # Add to allocations
                self.allocations.append(allocation)
                newly_allocated.append(vessel_id)

                infos[vessel_id] = {
                    'allocated': True,
                    'position': position,
                    'berthing_time': berthing_time,
                    'uses_shore_power': use_shore_power
                }
            else:
                # Invalid action - penalty
                reward = -self.config['rewards']['c8']
                infos[vessel_id] = {
                    'allocated': False,
                    'violation': violation_reason
                }

            rewards[vessel_id] = reward

        # Remove allocated vessels from pending
        for vessel_id in newly_allocated:
            self.pending_vessels.remove(vessel_id)

        # Update current time (advance to next vessel arrival or end)
        if self.pending_vessels:
            next_vessel = self.vessels[self.pending_vessels[0]]
            self.current_time = next_vessel.arrival_time
        else:
            self.current_time = self.planning_horizon_hours

        # Get next observations
        observations = self._get_observations()

        # Check termination
        terminated = len(self.pending_vessels) == 0
        truncated = self.episode_step >= self.max_episode_steps

        # Global info
        global_info = {
            'episode_step': self.episode_step,
            'vessels_allocated': len(self.allocations),
            'vessels_remaining': len(self.pending_vessels),
            'current_time': self.current_time
        }
        infos['_global'] = global_info

        return observations, rewards, terminated, truncated, infos

    def _check_constraints(self, vessel: Vessel, position: float,
                          berthing_time: float, departure_time: float,
                          use_shore_power: bool) -> Tuple[bool, Optional[str]]:
        """
        Check if allocation satisfies all constraints.

        Args:
            vessel: Vessel to allocate
            position: Berthing position
            berthing_time: Berthing time
            departure_time: Departure time
            use_shore_power: Whether using shore power

        Returns:
            (is_valid, violation_reason)
        """
        # 1. Boundary constraint
        if position < 0 or position + vessel.length > self.berth_length:
            return False, "boundary_violation"

        # 2. Time constraint (cannot berth before arrival)
        if berthing_time < vessel.arrival_time:
            return False, "time_violation"

        # 3. Spatial-temporal conflict with existing allocations
        for alloc in self.allocations:
            # Check time overlap
            time_overlap = not (departure_time <= alloc['berthing_time'] or
                              berthing_time >= alloc['departure_time'])

            if time_overlap:
                # Check spatial overlap
                other_start = alloc['position']
                other_end = alloc['position'] + alloc['vessel'].length

                vessel_start = position
                vessel_end = position + vessel.length

                # Add safe distance
                if not (vessel_end + self.safe_distance <= other_start or
                       vessel_start >= other_end + self.safe_distance):
                    return False, f"conflict_with_vessel_{alloc['vessel'].id}"

        # 4. Shore power constraint
        if use_shore_power:
            available, _ = self.shore_power.check_availability(
                position, vessel.length, vessel.power_requirement
            )
            if not available:
                return False, "shore_power_unavailable"

        return True, None

    def _allocate_berth(self, vessel: Vessel, position: float,
                       berthing_time: float, departure_time: float,
                       use_shore_power: bool) -> Dict:
        """
        Create allocation record.

        Args:
            vessel: Vessel
            position: Position
            berthing_time: Berthing time
            departure_time: Departure time
            use_shore_power: Whether using shore power

        Returns:
            Allocation dictionary
        """
        allocation = {
            'vessel': vessel,
            'vessel_id': vessel.id,
            'position': position,
            'berthing_time': berthing_time,
            'departure_time': departure_time,
            'uses_shore_power': use_shore_power,
            'shore_power_segments': []
        }

        # Allocate shore power if needed
        if use_shore_power:
            segment_ids = self.shore_power.find_covering_segments(position, vessel.length)
            success = self.shore_power.allocate_power(
                vessel.id, position, vessel.length,
                vessel.power_requirement, berthing_time, departure_time
            )
            if success:
                allocation['shore_power_segments'] = segment_ids

        return allocation

    def _calculate_reward(self, vessel: Vessel, allocation: Dict) -> float:
        """
        Calculate reward for an allocation.

        Note: Full reward calculation is delegated to RewardCalculator,
        but we implement a simple version here for standalone testing.

        Args:
            vessel: Vessel
            allocation: Allocation info

        Returns:
            Reward value
        """
        from rewards.reward_calculator import RewardCalculator

        calculator = RewardCalculator(self.config)
        reward = calculator.calculate(vessel, allocation, self._get_env_state())

        return reward

    def _get_observation(self, vessel_id: int) -> np.ndarray:
        """
        Get observation for a single vessel (agent).

        Returns 17-dimensional feature vector (NO noise dimension).

        Args:
            vessel_id: Vessel ID

        Returns:
            Observation array
        """
        if vessel_id >= len(self.vessels):
            # Return zero observation for invalid ID
            return np.zeros(17, dtype=np.float32)

        vessel = self.vessels[vessel_id]

        # Static features (4)
        static_features = np.array([
            vessel.length / self.berth_length,  # Normalized length
            vessel.arrival_time / self.planning_horizon_hours,  # Normalized arrival
            (vessel.priority - 1) / 3.0,  # Normalized priority (1-4 -> 0-1)
            float(vessel.can_use_shore_power)  # Boolean
        ], dtype=np.float32)

        # Dynamic features (3)
        waiting_time = max(0, self.current_time - vessel.arrival_time)
        dynamic_features = np.array([
            self.current_time / self.planning_horizon_hours,  # Normalized time
            waiting_time / 48.0,  # Normalized waiting (max 48h)
            vessel.operation_time / 48.0  # Normalized operation time
        ], dtype=np.float32)

        # Shore power features (6: 5 segments + total)
        shore_power_features = self.shore_power.get_state_features()

        # Berth dynamic features (4)
        berth_features = self._get_berth_features(vessel_id)

        # Concatenate all features (17 total)
        observation = np.concatenate([
            static_features,      # 4
            dynamic_features,     # 3
            shore_power_features, # 6
            berth_features        # 4
        ])

        return observation.astype(np.float32)

    def _get_berth_features(self, vessel_id: int) -> np.ndarray:
        """
        Get berth-related features for a vessel.

        Returns:
            Array of [left_neighbor_dist, right_neighbor_dist, available_left, available_right]
        """
        vessel = self.vessels[vessel_id]

        # Find neighboring allocated vessels at current time
        left_neighbor_dist = self.berth_length  # Max distance
        right_neighbor_dist = self.berth_length
        available_left = self.berth_length
        available_right = self.berth_length

        # Search through allocations that overlap with current time
        for alloc in self.allocations:
            # Check if allocation is active around current time
            if alloc['berthing_time'] <= self.current_time <= alloc['departure_time']:
                other_pos = alloc['position']
                other_len = alloc['vessel'].length

                # Estimate vessel's potential position (center of berth)
                vessel_pos_estimate = self.berth_length / 2

                # Left neighbor
                if other_pos + other_len < vessel_pos_estimate:
                    dist = vessel_pos_estimate - (other_pos + other_len)
                    left_neighbor_dist = min(left_neighbor_dist, dist)

                # Right neighbor
                if other_pos > vessel_pos_estimate:
                    dist = other_pos - vessel_pos_estimate
                    right_neighbor_dist = min(right_neighbor_dist, dist)

        # Available space
        available_left = min(available_left, left_neighbor_dist)
        available_right = min(available_right, right_neighbor_dist)

        # Normalize
        features = np.array([
            left_neighbor_dist / self.berth_length,
            right_neighbor_dist / self.berth_length,
            available_left / self.berth_length,
            available_right / self.berth_length
        ], dtype=np.float32)

        return features

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Get observations for all pending vessels.

        Returns:
            Dictionary mapping vessel_id to observation
        """
        observations = {}
        for vessel_id in self.pending_vessels:
            observations[vessel_id] = self._get_observation(vessel_id)
        return observations

    def _get_env_state(self) -> Dict:
        """
        Get current environment state for reward calculation.

        Returns:
            Environment state dictionary
        """
        return {
            'current_allocations': self.allocations,
            'current_time': self.current_time,
            'berth_length': self.berth_length,
            'shore_power_usage': self.shore_power.get_total_usage(),
            'shore_power_capacity': self.shore_power.get_total_capacity()
        }

    def get_global_state(self) -> np.ndarray:
        """
        Get global state for centralized critic (CTDE).

        Returns:
            Global state array (concatenation of all local observations)
        """
        observations = []
        for vessel_id in range(len(self.vessels)):
            obs = self._get_observation(vessel_id)
            observations.append(obs)

        # Pad if necessary
        while len(observations) < self.max_vessels:
            observations.append(np.zeros(17, dtype=np.float32))

        global_state = np.concatenate(observations)
        return global_state

    def render(self, mode='human'):
        """Render environment (placeholder for visualization)."""
        if mode == 'human':
            print(f"\n=== Berth Allocation State (t={self.current_time:.1f}h) ===")
            print(f"Allocated: {len(self.allocations)}/{len(self.vessels)}")
            print(f"Pending: {len(self.pending_vessels)}")
            print(f"Shore Power Usage: {self.shore_power.get_total_usage():.1f}/"
                  f"{self.shore_power.get_total_capacity():.1f} kW")

    def close(self):
        """Clean up resources."""
        pass
