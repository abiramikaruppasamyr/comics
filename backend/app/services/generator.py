from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import LORA_PATHS, get_settings
from app.schemas.generation import GenerateImageRequest, GenerateImageResponse, ImageResult
from app.services.pipeline import PipelineManager
from app.services.system_metrics import SystemMetricsService
from app.utils.seed import resolve_seed


class ImageGeneratorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline_manager = PipelineManager()
        self.system_metrics_service = SystemMetricsService()

    def generate(self, payload: GenerateImageRequest) -> GenerateImageResponse:
        self._validate_model_files()

        seed = resolve_seed(payload.seed)
        image = self.pipeline_manager.generate_image(
            positive_prompt=payload.positive_prompt,
            negative_prompt=payload.negative_prompt or self.settings.default_negative_prompt,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            cfg_scale=payload.cfg_scale,
            denoise_strength=payload.denoise_strength,
            seed=seed,
            lora_style=payload.lora_style,
            lora_strength=payload.lora_strength,
        )

        generated_at = datetime.now(timezone.utc)
        filename = self._build_filename(generated_at)
        output_path = self.settings.output_dir / filename
        image.save(output_path)

        system = self.system_metrics_service.snapshot()
        image_result = ImageResult(
            filename=filename,
            image_url=f"/output/{filename}",
            seed=seed,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            cfg_scale=payload.cfg_scale,
            positive_prompt=payload.positive_prompt,
            negative_prompt=payload.negative_prompt,
            generated_at=generated_at,
        )
        return GenerateImageResponse(image=image_result, system=system)

    def _validate_model_files(self) -> None:
        if not self.settings.model_path.exists():
            raise FileNotFoundError(f"Base model file not found: {self.settings.model_path}")
        if not self.settings.model_config_path.exists():
            raise FileNotFoundError(f"Model config YAML not found: {self.settings.model_config_path}")
        missing_loras = [path for path in LORA_PATHS.values() if not Path(path).exists()]
        if missing_loras:
            raise FileNotFoundError(f"LoRA file not found: {missing_loras[0]}")

    @staticmethod
    def _build_filename(timestamp: datetime) -> str:
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        return f"generated_{stamp}_{uuid4().hex[:8]}.png"
