"""Request models for API endpoints."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class TaskConfigRequest(BaseModel):
    """Task configuration request model."""

    num_vessels: int = Field(ge=5, le=50, description="Number of vessels to generate")
    planning_horizon: int = Field(ge=1, le=14, description="Planning horizon in days")
    generation_mode: Literal['realistic', 'simple'] = Field(
        default='realistic',
        description="Vessel generation mode"
    )
    berth_length: int = Field(default=2000, ge=1000, le=5000, description="Berth length in meters")
    shore_power_enabled: bool = Field(default=True, description="Enable shore power")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "num_vessels": 20,
                "planning_horizon": 7,
                "generation_mode": "realistic",
                "berth_length": 2000,
                "shore_power_enabled": True,
                "seed": 42
            }
        }


class AlgorithmRunRequest(BaseModel):
    """Algorithm execution request model."""

    task_id: str = Field(description="Task ID from task generation")
    algorithm: Literal['MATD3', 'Greedy', 'FCFS'] = Field(
        default='MATD3',
        description="Algorithm to use for allocation"
    )
    model_path: Optional[str] = Field(
        default='results/models/best_model.pth',
        description="Path to trained model (for MATD3)"
    )
    realtime: bool = Field(
        default=False,
        description="Use WebSocket for real-time streaming"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "uuid-example",
                "algorithm": "MATD3",
                "model_path": "results/models/best_model.pth",
                "realtime": False
            }
        }
