"""
Reward calculator for berth allocation with multi-objective optimization.

Implements hierarchical reward shaping for:
- Economic efficiency (waiting time, berth utilization)
- Environmental impact (carbon emissions, shore power usage)
- Operational constraints (spacing, conflicts)
"""

import numpy as np
from typing import Dict, Any, List


class RewardCalculator:
    """
    Multi-objective reward calculator.

    Based on paper design with improvements from reviewer feedback.
    """

    def __init__(self, config: dict):
        """
        Initialize reward calculator.

        Args:
            config: Configuration dict with:
                - rewards: Dict of c1-c8 weights
                - berth_length: Berth length
                - planning_horizon: Planning horizon (days)
                - max_wait_time: Maximum waiting time (hours)
                - safe_distance: Safe distance between vessels
                - shore_power: Shore power config
        """
        self.config = config
        self.rewards_config = config.get('rewards', {})

        # Reward weights (c1-c8)
        self.c1 = self.rewards_config.get('c1', 10.0)  # Base positive reward
        self.c2 = self.rewards_config.get('c2', 5.0)   # Waiting time penalty
        self.c3 = self.rewards_config.get('c3', 8.0)   # Emission penalty
        self.c4 = self.rewards_config.get('c4', 6.0)   # Shore power bonus
        self.c5 = self.rewards_config.get('c5', 4.0)   # Berth utilization reward
        self.c6 = self.rewards_config.get('c6', 3.0)   # Spacing reward (improved)
        self.c7 = self.rewards_config.get('c7', 0.0)   # Reserved
        self.c8 = self.rewards_config.get('c8', 20.0)  # Invalid action penalty

        # Environment parameters
        self.berth_length = config['berth_length']
        self.planning_horizon = config['planning_horizon'] * 24  # Convert to hours
        self.max_wait_time = config.get('max_wait_time', 48.0)
        self.safe_distance = config.get('safe_distance', 20.0)

        # Emission factors
        shore_power_config = config.get('shore_power', {})
        self.emission_factor_ship = shore_power_config.get('emission_factor_ship', 2500)
        self.emission_factor_shore = shore_power_config.get('emission_factor_shore', 800)

    def calculate(self, vessel: Any, allocation: Dict, env_state: Dict) -> float:
        """
        Calculate total reward for an allocation.

        Args:
            vessel: Vessel object
            allocation: Allocation dict with:
                - position: Berth position
                - berthing_time: Berthing time
                - departure_time: Departure time
                - uses_shore_power: Whether using shore power
            env_state: Environment state dict

        Returns:
            Total reward
        """
        reward = 0.0

        # (1) Base positive reward (priority-dependent)
        reward += self.c1 * self._base_reward(vessel)

        # (2) Waiting time penalty
        waiting_time = allocation['berthing_time'] - vessel.arrival_time
        reward -= self.c2 * self._waiting_penalty(vessel, waiting_time)

        # (3) Emission penalty
        reward -= self.c3 * self._emission_penalty(
            vessel, allocation['uses_shore_power']
        )

        # (4) Shore power usage bonus
        if allocation['uses_shore_power']:
            reward += self.c4

        # (5) Berth utilization reward
        reward += self.c5 * self._berth_utilization_reward(env_state)

        # (6) Spacing reward (improved - based on neighbor crowding)
        reward += self.c6 * self._spacing_reward(
            allocation['position'], vessel.length, env_state
        )

        return reward

    def _base_reward(self, vessel: Any) -> float:
        """
        Base positive reward based on vessel priority.

        Higher priority vessels get higher base reward.

        Args:
            vessel: Vessel object

        Returns:
            Base reward (normalized to ~1.0)
        """
        priority_rewards = {
            1: 1.0,   # Highest priority
            2: 0.7,
            3: 0.5,
            4: 0.3    # Lowest priority
        }
        return priority_rewards.get(vessel.priority, 0.5)

    def _waiting_penalty(self, vessel: Any, wait_time: float) -> float:
        """
        Waiting time penalty with priority consideration.

        Higher priority vessels receive heavier penalties for waiting.

        Args:
            vessel: Vessel object
            wait_time: Waiting time in hours

        Returns:
            Normalized penalty (0-1)
        """
        # Normalize waiting time
        normalized_wait = wait_time / self.max_wait_time
        normalized_wait = np.clip(normalized_wait, 0, 1)

        # Priority multiplier
        priority_multipliers = {
            1: 2.0,   # Double penalty for highest priority
            2: 1.5,
            3: 1.0,
            4: 0.5    # Half penalty for lowest priority
        }
        multiplier = priority_multipliers.get(vessel.priority, 1.0)

        # Quadratic penalty (penalize long waits more)
        penalty = multiplier * (normalized_wait ** 2)

        return penalty

    def _emission_penalty(self, vessel: Any, uses_shore_power: bool) -> float:
        """
        Carbon emission penalty.

        Args:
            vessel: Vessel object
            uses_shore_power: Whether using shore power

        Returns:
            Normalized emission penalty (0-1)
        """
        # Calculate energy consumption (MWh)
        energy_mwh = (vessel.power_requirement * vessel.operation_time) / 1000

        # Calculate emissions based on power source
        if uses_shore_power:
            emissions = energy_mwh * self.emission_factor_shore
        else:
            emissions = energy_mwh * self.emission_factor_ship

        # Normalize (assume max vessel: 1000kW * 48h * ship factor)
        max_emissions = (1000 * 48 / 1000) * self.emission_factor_ship
        normalized_emissions = emissions / max_emissions

        return np.clip(normalized_emissions, 0, 1)

    def _berth_utilization_reward(self, env_state: Dict) -> float:
        """
        Berth utilization reward.

        Rewards achieving target utilization rate (e.g., 70%).

        Args:
            env_state: Environment state

        Returns:
            Utilization reward (0-1)
        """
        allocations = env_state.get('current_allocations', [])

        if not allocations:
            return 0.0

        # Calculate occupied berth length
        occupied_length = sum(
            alloc['vessel'].length
            for alloc in allocations
        )

        # Calculate utilization ratio
        utilization = occupied_length / self.berth_length

        # Target utilization (configurable, default 70%)
        target_utilization = self.config.get('target_utilization', 0.7)

        # Reward proximity to target
        deviation = abs(utilization - target_utilization)
        reward = max(0, 1.0 - deviation)

        return reward

    def _spacing_reward(self, position: float, length: float,
                       env_state: Dict) -> float:
        """
        Spacing reward based on neighbor crowding (IMPROVED).

        Improvement from reviewer feedback:
        - No longer biased toward berth center
        - Based on distance to nearest neighbors
        - Uses convex penalty for close spacing

        Args:
            position: Vessel position
            length: Vessel length
            env_state: Environment state

        Returns:
            Spacing reward (-1 to 1)
        """
        allocations = env_state.get('current_allocations', [])

        # Find neighbor distances
        neighbor_distances = self._find_neighbor_distances(
            position, length, allocations
        )

        if not neighbor_distances:
            # No neighbors, moderate reward
            return 0.5

        # Get minimum distance to any neighbor
        min_distance = min(neighbor_distances)

        # Reward based on distance
        if min_distance < self.safe_distance:
            # Too close to safe distance - penalty
            penalty_ratio = (self.safe_distance - min_distance) / self.safe_distance
            reward = -1.0 * penalty_ratio
        else:
            # Good spacing - reward
            # Saturate at 100m beyond safe distance
            reward = min(1.0, (min_distance - self.safe_distance) / 100.0)

        return reward

    def _find_neighbor_distances(self, position: float, length: float,
                                 allocations: List[Dict]) -> List[float]:
        """
        Find distances to neighboring vessels.

        Args:
            position: Current vessel position
            length: Current vessel length
            allocations: List of existing allocations

        Returns:
            List of distances to neighbors
        """
        distances = []
        vessel_end = position + length

        for alloc in allocations:
            other_pos = alloc['position']
            other_len = alloc['vessel'].length
            other_end = other_pos + other_len

            # Calculate spacing
            if other_end <= position:
                # Neighbor on the left
                distance = position - other_end
            elif other_pos >= vessel_end:
                # Neighbor on the right
                distance = other_pos - vessel_end
            else:
                # Overlap (should not happen in valid allocations)
                distance = 0.0

            distances.append(distance)

        return distances

    def calculate_metrics(self, allocations: List[Dict]) -> Dict[str, float]:
        """
        Calculate aggregate metrics for evaluation.

        Args:
            allocations: List of all allocations

        Returns:
            Dictionary of metrics
        """
        if not allocations:
            return {
                'avg_waiting_time': 0.0,
                'total_emissions': 0.0,
                'berth_utilization': 0.0,
                'shore_power_usage_rate': 0.0,
                'num_vessels': 0
            }

        # Average waiting time
        waiting_times = [
            alloc['berthing_time'] - alloc['vessel'].arrival_time
            for alloc in allocations
        ]
        avg_waiting_time = np.mean(waiting_times)

        # Total emissions
        total_emissions = sum(
            self._calculate_emissions(
                alloc['vessel'],
                alloc['uses_shore_power']
            )
            for alloc in allocations
        )

        # Berth utilization (space-time)
        total_berth_time = sum(
            (alloc['departure_time'] - alloc['berthing_time']) * alloc['vessel'].length
            for alloc in allocations
        )
        max_berth_time = self.berth_length * self.planning_horizon
        berth_utilization = total_berth_time / max_berth_time

        # Shore power usage rate
        shore_power_count = sum(1 for alloc in allocations if alloc['uses_shore_power'])
        shore_power_capable = sum(
            1 for alloc in allocations if alloc['vessel'].can_use_shore_power
        )
        shore_power_usage_rate = (
            shore_power_count / shore_power_capable if shore_power_capable > 0 else 0.0
        )

        return {
            'avg_waiting_time': avg_waiting_time,
            'total_emissions': total_emissions,
            'berth_utilization': berth_utilization,
            'shore_power_usage_rate': shore_power_usage_rate,
            'num_vessels': len(allocations)
        }

    def _calculate_emissions(self, vessel: Any, uses_shore_power: bool) -> float:
        """
        Calculate emissions for a single vessel.

        Args:
            vessel: Vessel object
            uses_shore_power: Whether using shore power

        Returns:
            Emissions in kg CO2
        """
        energy_mwh = (vessel.power_requirement * vessel.operation_time) / 1000

        if uses_shore_power:
            return energy_mwh * self.emission_factor_shore
        else:
            return energy_mwh * self.emission_factor_ship
