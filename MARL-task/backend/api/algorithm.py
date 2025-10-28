"""Algorithm execution API endpoints."""

from fastapi import APIRouter, HTTPException
import uuid

from backend.models import AlgorithmRunRequest, AllocationResponse
from backend.services import task_manager, algorithm_runner, result_cache

router = APIRouter()


@router.post("/run", response_model=AllocationResponse)
async def run_algorithm(request: AlgorithmRunRequest):
    """
    Run berth allocation algorithm.

    Executes specified algorithm on task and returns allocation results.

    Args:
        request: Algorithm execution request

    Returns:
        Allocation results with metrics

    Raises:
        HTTPException: If task not found or algorithm fails
    """
    # Verify task exists
    task_data = task_manager.get_task(request.task_id)

    if not task_data:
        raise HTTPException(
            status_code=404,
            detail=f"Task {request.task_id} not found"
        )

    try:
        allocation_id = str(uuid.uuid4())

        # Run algorithm
        result = algorithm_runner.run(
            task_data=task_data,
            algorithm=request.algorithm,
            model_path=request.model_path
        )

        # Cache result
        result_cache.set(allocation_id, result)

        # Return response
        return AllocationResponse(
            allocation_id=allocation_id,
            allocations=result['allocations'],
            metrics=result['metrics']
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Algorithm execution failed: {str(e)}"
        )


@router.get("/result/{allocation_id}", response_model=AllocationResponse)
async def get_result(allocation_id: str):
    """
    Retrieve allocation result by ID.

    Args:
        allocation_id: Allocation identifier

    Returns:
        Allocation results with metrics

    Raises:
        HTTPException: If result not found
    """
    result = result_cache.get(allocation_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Result {allocation_id} not found"
        )

    return AllocationResponse(
        allocation_id=allocation_id,
        allocations=result['allocations'],
        metrics=result['metrics']
    )
