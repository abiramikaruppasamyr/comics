import psutil

from app.schemas.system import SystemMetricsResponse


class SystemMetricsService:
    def snapshot(self) -> SystemMetricsResponse:
        memory = psutil.virtual_memory()
        return SystemMetricsResponse(
            cpu_percent=round(psutil.cpu_percent(interval=0.2), 1),
            memory_percent=round(memory.percent, 1),
            memory_used_mb=round(memory.used / (1024 * 1024), 1),
            memory_available_mb=round(memory.available / (1024 * 1024), 1),
        )
