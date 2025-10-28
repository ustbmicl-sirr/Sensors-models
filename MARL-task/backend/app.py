"""
FastAPI application for Berth Allocation Visualization.

This is the main backend server that provides REST API and WebSocket endpoints
for the berth allocation visualization web application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api import task_router, algorithm_router, websocket_router
from backend.services import result_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("="*60)
    print("🚀 Starting Berth Allocation Visualization API Server")
    print("="*60)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    print("="*60)

    yield

    # Shutdown
    print("\n" + "="*60)
    print("🛑 Shutting down server...")
    result_cache.clear()
    print("✓ Cleanup complete")
    print("="*60)


# Create FastAPI application
app = FastAPI(
    title="Berth Allocation Visualization API",
    description="API for visualizing MATD3-based berth allocation and shore power coordination",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(
    task_router,
    prefix="/api/task",
    tags=["Task Management"]
)

app.include_router(
    algorithm_router,
    prefix="/api/algorithm",
    tags=["Algorithm Execution"]
)

app.include_router(
    websocket_router,
    tags=["WebSocket"]
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Berth Allocation Visualization API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "generate_task": "POST /api/task/generate",
            "get_task": "GET /api/task/{task_id}",
            "run_algorithm": "POST /api/algorithm/run",
            "get_result": "GET /api/algorithm/result/{allocation_id}",
            "websocket_stream": "WS /ws/stream"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "berth-allocation-api",
        "cache_size": len(result_cache)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
