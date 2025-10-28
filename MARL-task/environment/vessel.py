"""
Vessel class and vessel generator for berth allocation environment.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class Vessel:
    """Vessel data class representing a ship in the port."""

    id: int
    length: float           # Ship length in meters
    draft: float            # Draft in meters
    arrival_time: float     # Arrival time in hours from planning start
    operation_time: float   # Required operation time in hours
    priority: int           # Priority level (1=highest, 4=lowest)
    can_use_shore_power: bool  # Whether ship can use shore power
    power_requirement: float   # Power requirement in kW

    # Runtime attributes
    berthing_time: Optional[float] = None      # Actual berthing time
    departure_time: Optional[float] = None     # Departure time
    position: Optional[float] = None           # Berthing position
    uses_shore_power: bool = False             # Whether actually using shore power
    waiting_time: Optional[float] = None       # Actual waiting time

    def __repr__(self):
        return (f"Vessel(id={self.id}, length={self.length:.1f}m, "
                f"arrival={self.arrival_time:.1f}h, priority={self.priority})")


class VesselGenerator:
    """
    Vessel generator with realistic arrival patterns and ship characteristics.

    Improvements over simple uniform distribution:
    - Non-homogeneous Poisson process for arrival times
    - Multi-modal distributions for vessel sizes
    - Peak hours simulation
    - Correlation between vessel attributes
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        """
        Initialize vessel generator.

        Args:
            config: Configuration dictionary with generation parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate_realistic(self, num_vessels: int, planning_horizon_days: int) -> List[Vessel]:
        """
        Generate realistic vessel data using non-homogeneous Poisson process.

        Args:
            num_vessels: Number of vessels to generate
            planning_horizon_days: Planning horizon in days

        Returns:
            List of Vessel objects
        """
        planning_horizon_hours = planning_horizon_days * 24

        # Generate arrival times with peak hours
        arrival_times = self._generate_arrival_times(
            num_vessels,
            planning_horizon_hours,
            peak_hours=self.config.get('peak_hours', [6, 12, 18]),
            peak_rate=self.config.get('peak_rate', 2.0)
        )

        vessels = []
        for i in range(num_vessels):
            vessel = self._generate_single_vessel(i, arrival_times[i])
            vessels.append(vessel)

        return vessels

    def generate_simple(self, num_vessels: int, planning_horizon_days: int) -> List[Vessel]:
        """
        Generate vessels with simple uniform distribution (baseline).

        Args:
            num_vessels: Number of vessels to generate
            planning_horizon_days: Planning horizon in days

        Returns:
            List of Vessel objects
        """
        planning_horizon_hours = planning_horizon_days * 24
        vessels = []

        for i in range(num_vessels):
            # Simple uniform arrival times
            arrival_time = self.rng.uniform(0, planning_horizon_hours / 6)

            # Simple uniform vessel length
            length = self.rng.uniform(80, 400)

            # Draft proportional to length
            draft = length * 0.05

            # Simple operation time
            operation_time = self.rng.uniform(4, 48)

            # Random priority
            priority = self.rng.choice([1, 2, 3, 4])

            # 50% can use shore power
            can_use_shore = self.rng.random() < 0.5

            # Power requirement
            power_requirement = self.rng.uniform(100, 1000)

            vessel = Vessel(
                id=i,
                length=length,
                draft=draft,
                arrival_time=arrival_time,
                operation_time=operation_time,
                priority=priority,
                can_use_shore_power=can_use_shore,
                power_requirement=power_requirement
            )
            vessels.append(vessel)

        return vessels

    def _generate_arrival_times(self, num_vessels: int, planning_horizon: float,
                                peak_hours: List[int], peak_rate: float) -> np.ndarray:
        """
        Generate arrival times using non-homogeneous Poisson process.

        Args:
            num_vessels: Number of vessels
            planning_horizon: Planning horizon in hours
            peak_hours: List of peak hours in day (0-23)
            peak_rate: Rate multiplier during peak hours

        Returns:
            Array of sorted arrival times
        """
        times = []
        current_time = 0.0
        base_rate = 1.0

        while len(times) < num_vessels and current_time < planning_horizon:
            # Calculate current rate based on time of day
            hour_of_day = current_time % 24

            # Check if in peak period (within 2 hours of peak)
            is_peak = any(abs(hour_of_day - ph) < 2 for ph in peak_hours)
            current_rate = peak_rate * base_rate if is_peak else base_rate

            # Generate inter-arrival time
            interval = self.rng.exponential(1.0 / current_rate)
            current_time += interval

            if current_time < planning_horizon:
                times.append(current_time)

        # Sort and return only required number
        times = sorted(times)[:num_vessels]

        # If not enough, fill with uniform distribution
        while len(times) < num_vessels:
            times.append(self.rng.uniform(0, planning_horizon))

        return np.array(sorted(times))

    def _generate_single_vessel(self, vessel_id: int, arrival_time: float) -> Vessel:
        """
        Generate a single vessel with realistic correlated attributes.

        Args:
            vessel_id: Unique vessel ID
            arrival_time: Arrival time in hours

        Returns:
            Vessel object
        """
        # Vessel size with multi-modal distribution
        vessel_type = self.rng.choice(
            ['small', 'medium', 'large'],
            p=self.config.get('size_distribution', [0.3, 0.5, 0.2])
        )

        if vessel_type == 'small':
            length = self.rng.normal(120, 20)
        elif vessel_type == 'medium':
            length = self.rng.normal(200, 30)
        else:  # large
            length = self.rng.normal(300, 40)

        # Clip to valid range
        length = np.clip(length, 80, 400)

        # Draft correlated with length
        draft = length * 0.05 + self.rng.normal(0, 0.5)
        draft = np.clip(draft, 5, 20)

        # Operation time correlated with length
        # Larger ships need more time
        operation_time = length / 10 + self.rng.exponential(5)
        operation_time = np.clip(operation_time, 4, 48)

        # Priority: larger ships tend to have higher priority
        if length > 250:
            priority = self.rng.choice([1, 2], p=[0.7, 0.3])
        elif length > 150:
            priority = self.rng.choice([2, 3], p=[0.6, 0.4])
        else:
            priority = self.rng.choice([3, 4], p=[0.5, 0.5])

        # Shore power availability (60% of modern vessels)
        can_use_shore = self.rng.random() < self.config.get('shore_power_ratio', 0.6)

        # Power requirement correlated with length
        power_requirement = length * 2 + self.rng.normal(0, 50)
        power_requirement = np.clip(power_requirement, 100, 1000)

        return Vessel(
            id=vessel_id,
            length=length,
            draft=draft,
            arrival_time=arrival_time,
            operation_time=operation_time,
            priority=priority,
            can_use_shore_power=can_use_shore,
            power_requirement=power_requirement
        )

    def load_from_csv(self, filepath: str) -> List[Vessel]:
        """
        Load real vessel data from CSV file.

        Expected CSV columns:
        - id, length, draft, arrival_time, operation_time, priority,
          can_use_shore_power, power_requirement

        Args:
            filepath: Path to CSV file

        Returns:
            List of Vessel objects
        """
        df = pd.read_csv(filepath)

        vessels = []
        for idx, row in df.iterrows():
            vessel = Vessel(
                id=int(row.get('id', idx)),
                length=float(row['length']),
                draft=float(row['draft']),
                arrival_time=float(row['arrival_time']),
                operation_time=float(row['operation_time']),
                priority=int(row['priority']),
                can_use_shore_power=bool(row['can_use_shore_power']),
                power_requirement=float(row['power_requirement'])
            )
            vessels.append(vessel)

        return vessels

    def save_to_csv(self, vessels: List[Vessel], filepath: str):
        """
        Save vessel data to CSV file.

        Args:
            vessels: List of vessels
            filepath: Output file path
        """
        data = []
        for v in vessels:
            data.append({
                'id': v.id,
                'length': v.length,
                'draft': v.draft,
                'arrival_time': v.arrival_time,
                'operation_time': v.operation_time,
                'priority': v.priority,
                'can_use_shore_power': v.can_use_shore_power,
                'power_requirement': v.power_requirement
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
