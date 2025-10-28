"""Response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import List


class VesselData(BaseModel):
    """Vessel data model."""

    id: int
    length: float = Field(description="Vessel length in meters")
    draft: float = Field(description="Draft in meters")
    arrival_time: float = Field(description="Arrival time in hours")
    operation_time: float = Field(description="Operation time in hours")
    priority: int = Field(ge=1, le=4, description="Priority level (1=highest)")
    can_use_shore_power: bool
    power_requirement: float = Field(description="Power requirement in kW")


class ShorePowerSegmentData(BaseModel):
    """Shore power segment data model."""

    id: int
    start: float = Field(description="Start position in meters")
    end: float = Field(description="End position in meters")
    capacity: float = Field(description="Capacity in kW")


class EnvironmentData(BaseModel):
    """Environment data model."""

    berth_length: float = Field(description="Total berth length in meters")
    shore_power_segments: List[ShorePowerSegmentData]


class TaskResponse(BaseModel):
    """Task generation response model."""

    task_id: str
    vessels: List[VesselData]
    environment: EnvironmentData

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "uuid-example",
                "vessels": [
                    {
                        "id": 0,
                        "length": 180.5,
                        "draft": 9.2,
                        "arrival_time": 3.5,
                        "operation_time": 12.0,
                        "priority": 1,
                        "can_use_shore_power": True,
                        "power_requirement": 450.0
                    }
                ],
                "environment": {
                    "berth_length": 2000,
                    "shore_power_segments": []
                }
            }
        }


class AllocationData(BaseModel):
    """Allocation result data model."""

    vessel_id: int
    position: float = Field(description="Berth position in meters")
    berthing_time: float = Field(description="Berthing time in hours")
    departure_time: float = Field(description="Departure time in hours")
    uses_shore_power: bool
    waiting_time: float = Field(description="Waiting time in hours")


class MetricsData(BaseModel):
    """Performance metrics data model."""

    berth_utilization: float = Field(description="Berth utilization ratio [0,1]")
    avg_waiting_time: float = Field(description="Average waiting time in hours")
    total_emissions: float = Field(description="Total emissions in kg CO2")
    shore_power_usage_rate: float = Field(description="Shore power usage rate [0,1]")
    num_vessels: int = Field(description="Number of vessels allocated")


class AllocationResponse(BaseModel):
    """Allocation result response model."""

    allocation_id: str
    allocations: List[AllocationData]
    metrics: MetricsData

    class Config:
        json_schema_extra = {
            "example": {
                "allocation_id": "uuid-example",
                "allocations": [
                    {
                        "vessel_id": 0,
                        "position": 120.5,
                        "berthing_time": 5.2,
                        "departure_time": 17.2,
                        "uses_shore_power": True,
                        "waiting_time": 1.7
                    }
                ],
                "metrics": {
                    "berth_utilization": 0.68,
                    "avg_waiting_time": 2.3,
                    "total_emissions": 15420.5,
                    "shore_power_usage_rate": 0.75,
                    "num_vessels": 20
                }
            }
        }
