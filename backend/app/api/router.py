from fastapi import APIRouter

from app.api.routes import generation, health, system


api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(generation.router, prefix="/generation", tags=["generation"])
