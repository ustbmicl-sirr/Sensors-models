"""
Shore power management system for berth allocation environment.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class ShorePowerSegment:
    """Single shore power segment with capacity and coverage."""

    def __init__(self, segment_id: int, start: float, end: float, capacity: float):
        """
        Initialize shore power segment.

        Args:
            segment_id: Unique segment ID
            start: Start position along berth (meters)
            end: End position along berth (meters)
            capacity: Maximum power capacity (kW)
        """
        self.id = segment_id
        self.start = start
        self.end = end
        self.capacity = capacity
        self.current_usage = 0.0
        self.allocations: List[Dict] = []  # List of current allocations

    def covers_position(self, position: float, length: float) -> bool:
        """
        Check if segment covers a vessel at given position.

        Args:
            position: Vessel position (bow)
            length: Vessel length

        Returns:
            True if segment covers any part of the vessel
        """
        vessel_start = position
        vessel_end = position + length

        # Check overlap
        return not (vessel_end <= self.start or vessel_start >= self.end)

    def available_capacity(self) -> float:
        """Get available capacity."""
        return self.capacity - self.current_usage

    def can_allocate(self, power: float) -> bool:
        """Check if power can be allocated."""
        return self.current_usage + power <= self.capacity

    def allocate(self, vessel_id: int, power: float, start_time: float, end_time: float):
        """
        Allocate power to a vessel.

        Args:
            vessel_id: Vessel ID
            power: Power to allocate (kW)
            start_time: Start time
            end_time: End time
        """
        if not self.can_allocate(power):
            raise ValueError(f"Insufficient capacity in segment {self.id}")

        self.current_usage += power
        self.allocations.append({
            'vessel_id': vessel_id,
            'power': power,
            'start_time': start_time,
            'end_time': end_time
        })

    def release(self, vessel_id: int):
        """
        Release power allocation for a vessel.

        Args:
            vessel_id: Vessel ID
        """
        for alloc in self.allocations:
            if alloc['vessel_id'] == vessel_id:
                self.current_usage -= alloc['power']
                self.allocations.remove(alloc)
                break

    def get_usage_ratio(self) -> float:
        """Get current usage ratio (0-1)."""
        return self.current_usage / self.capacity if self.capacity > 0 else 0.0

    def __repr__(self):
        return (f"ShorePowerSegment(id={self.id}, "
                f"position={self.start}-{self.end}m, "
                f"usage={self.current_usage:.1f}/{self.capacity:.1f}kW)")


class ShorePowerManager:
    """
    Shore power management system.

    Manages multiple shore power segments along the berth,
    handles allocation, capacity checking, and emission calculations.
    """

    def __init__(self, config: dict):
        """
        Initialize shore power manager.

        Args:
            config: Configuration dictionary with:
                - segments: List of segment configs
                  [{start, end, capacity}, ...]
                - emission_factor_ship: Ship auxiliary engine (kg CO2/MWh)
                - emission_factor_shore: Shore power (kg CO2/MWh)
        """
        self.config = config

        # Create segments
        self.segments: List[ShorePowerSegment] = []
        for i, seg_config in enumerate(config['segments']):
            segment = ShorePowerSegment(
                segment_id=i,
                start=seg_config['start'],
                end=seg_config['end'],
                capacity=seg_config['capacity']
            )
            self.segments.append(segment)

        # Emission factors
        self.emission_factor_ship = config.get('emission_factor_ship', 2500)  # kg/MWh
        self.emission_factor_shore = config.get('emission_factor_shore', 800)  # kg/MWh

    def find_covering_segments(self, position: float, length: float) -> List[int]:
        """
        Find segments that cover a vessel position.

        Args:
            position: Vessel bow position
            length: Vessel length

        Returns:
            List of segment IDs that cover the vessel
        """
        covering = []
        for segment in self.segments:
            if segment.covers_position(position, length):
                covering.append(segment.id)
        return covering

    def check_availability(self, position: float, length: float,
                          required_power: float) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if shore power is available for a vessel.

        Args:
            position: Vessel bow position
            length: Vessel length
            required_power: Required power (kW)

        Returns:
            Tuple of (is_available, segment_ids)
        """
        # Find covering segments
        covering_segments = self.find_covering_segments(position, length)

        if not covering_segments:
            return False, None

        # Check if all covering segments have capacity
        for seg_id in covering_segments:
            if not self.segments[seg_id].can_allocate(required_power):
                return False, None

        return True, covering_segments

    def allocate_power(self, vessel_id: int, position: float, length: float,
                      power: float, start_time: float, end_time: float) -> bool:
        """
        Allocate shore power to a vessel.

        Args:
            vessel_id: Vessel ID
            position: Vessel position
            length: Vessel length
            power: Power requirement (kW)
            start_time: Start time
            end_time: End time

        Returns:
            True if allocation successful
        """
        available, segment_ids = self.check_availability(position, length, power)

        if not available:
            return False

        # Allocate to all covering segments
        for seg_id in segment_ids:
            self.segments[seg_id].allocate(vessel_id, power, start_time, end_time)

        return True

    def release_power(self, vessel_id: int, segment_ids: List[int]):
        """
        Release shore power allocation.

        Args:
            vessel_id: Vessel ID
            segment_ids: Segments to release from
        """
        for seg_id in segment_ids:
            self.segments[seg_id].release(vessel_id)

    def calculate_emissions(self, use_shore_power: bool, power: float,
                           duration: float) -> float:
        """
        Calculate carbon emissions.

        Args:
            use_shore_power: Whether using shore power
            power: Power consumption (kW)
            duration: Operation duration (hours)

        Returns:
            Total emissions (kg CO2)
        """
        # Convert to MWh
        energy_mwh = (power * duration) / 1000

        if use_shore_power:
            emissions = energy_mwh * self.emission_factor_shore
        else:
            emissions = energy_mwh * self.emission_factor_ship

        return emissions

    def get_state_features(self) -> np.ndarray:
        """
        Get shore power state features for observation.

        Returns:
            Array of features: [seg1_ratio, seg2_ratio, ..., total_ratio]
        """
        features = []

        # Usage ratio for each segment
        for segment in self.segments:
            features.append(segment.get_usage_ratio())

        # Total usage ratio
        total_usage = sum(seg.current_usage for seg in self.segments)
        total_capacity = sum(seg.capacity for seg in self.segments)
        total_ratio = total_usage / total_capacity if total_capacity > 0 else 0.0
        features.append(total_ratio)

        return np.array(features, dtype=np.float32)

    def get_total_capacity(self) -> float:
        """Get total shore power capacity."""
        return sum(seg.capacity for seg in self.segments)

    def get_total_usage(self) -> float:
        """Get total current usage."""
        return sum(seg.current_usage for seg in self.segments)

    def reset(self):
        """Reset all allocations."""
        for segment in self.segments:
            segment.current_usage = 0.0
            segment.allocations = []

    def get_usage_timeseries(self, time_points: np.ndarray,
                            allocations: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Get usage timeseries for visualization.

        Args:
            time_points: Array of time points
            allocations: List of allocation dictionaries

        Returns:
            Dict mapping segment_id to usage array
        """
        usage = {seg.id: np.zeros_like(time_points) for seg in self.segments}

        for alloc in allocations:
            if alloc.get('uses_shore_power'):
                start = alloc['berthing_time']
                end = alloc['departure_time']
                segments = alloc['shore_power_segments']
                power = alloc['vessel'].power_requirement

                # Find time interval
                mask = (time_points >= start) & (time_points <= end)

                # Add to usage
                for seg_id in segments:
                    usage[seg_id][mask] += power

        return usage

    def __repr__(self):
        return (f"ShorePowerManager(segments={len(self.segments)}, "
                f"total_capacity={self.get_total_capacity():.1f}kW, "
                f"usage={self.get_total_usage():.1f}kW)")
