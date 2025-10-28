"""Task management API endpoints."""

from fastapi import APIRouter, HTTPException
import uuid

from backend.models import TaskConfigRequest, TaskResponse
from backend.services import task_manager

router = APIRouter()


@router.post("/generate", response_model=TaskResponse)
async def generate_task(config: TaskConfigRequest):
    """
    Generate a new berth allocation task.

    Creates environment and generates vessels according to configuration.

    Args:
        config: Task configuration

    Returns:
        Generated task with ID, vessels, and environment data

    Raises:
        HTTPException: If task generation fails
    """
    try:
        task_id = str(uuid.uuid4())

        # Create task
        task_data = task_manager.create_task(task_id, config.dict())

        # Return response
        return TaskResponse(
            task_id=task_id,
            vessels=task_data['vessels'],
            environment=task_data['environment']
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate task: {str(e)}"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Retrieve an existing task by ID.

    Args:
        task_id: Task identifier

    Returns:
        Task data

    Raises:
        HTTPException: If task not found
    """
    task_data = task_manager.get_task(task_id)

    if not task_data:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    return TaskResponse(
        task_id=task_id,
        vessels=task_data['vessels'],
        environment=task_data['environment']
    )
