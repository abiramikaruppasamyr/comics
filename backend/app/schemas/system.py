from pydantic import BaseModel


class SystemMetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
