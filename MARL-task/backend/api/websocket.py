"""WebSocket API for real-time algorithm streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio

from backend.services import task_manager, algorithm_runner

router = APIRouter()


@router.websocket("/ws/stream")
async def algorithm_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time algorithm execution streaming.

    Client sends configuration, server streams allocation events and metrics updates.

    Protocol:
        Client -> Server: { "task_id": "...", "algorithm": "MATD3", "model_path": "..." }
        Server -> Client: { "type": "allocation", "vessel_id": ..., "position": ..., ... }
        Server -> Client: { "type": "metrics_update", "metrics": {...} }
        Server -> Client: { "type": "complete" }
        Server -> Client: { "type": "error", "message": "..." }
    """
    await websocket.accept()

    try:
        # Receive configuration
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)

        task_id = config.get('task_id')
        algorithm = config.get('algorithm', 'MATD3')
        model_path = config.get('model_path')

        # Validate task
        task_data = task_manager.get_task(task_id)

        if not task_data:
            await websocket.send_json({
                'type': 'error',
                'message': f'Task {task_id} not found'
            })
            await websocket.close()
            return

        # Stream algorithm execution
        async for message in algorithm_runner.run_streaming(
            task_data=task_data,
            algorithm=algorithm,
            model_path=model_path
        ):
            await websocket.send_json(message)
            await asyncio.sleep(0.05)  # Control streaming speed

        # Send completion message
        await websocket.send_json({'type': 'complete'})

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")

    except Exception as e:
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except:
            pass

    finally:
        try:
            await websocket.close()
        except:
            pass
