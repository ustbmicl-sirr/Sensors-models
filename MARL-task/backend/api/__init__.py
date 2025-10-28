"""API routers."""

from .task import router as task_router
from .algorithm import router as algorithm_router
from .websocket import router as websocket_router

__all__ = ['task_router', 'algorithm_router', 'websocket_router']
