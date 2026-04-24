from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.system import SystemMetricsResponse


class GenerateImageRequest(BaseModel):
    positive_prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    steps: int = Field(default=20, ge=1, le=100)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    seed: int | None = Field(default=None)

    @field_validator("width", "height")
    @classmethod
    def validate_dimension_multiple(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("Width and height must be divisible by 8.")
        return value


class ImageResult(BaseModel):
    filename: str
    image_url: str
    seed: int
    width: int
    height: int
    steps: int
    cfg_scale: float
    positive_prompt: str
    negative_prompt: str
    generated_at: datetime


class GenerateImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image: ImageResult
    system: SystemMetricsResponse
    message: str = "Image generated successfully."
