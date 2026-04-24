from fastapi import APIRouter

from app.schemas.system import SystemMetricsResponse
from app.services.system_metrics import SystemMetricsService


router = APIRouter()
system_metrics_service = SystemMetricsService()


@router.get("/metrics", response_model=SystemMetricsResponse)
def get_metrics() -> SystemMetricsResponse:
    return system_metrics_service.snapshot()
