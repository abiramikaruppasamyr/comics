from __future__ import annotations

import copy
import gc
import threading
from typing import Any

import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from app.core.config import LORA_PATHS, get_settings
from app.services.lora_loader import load_a1111_lora_into_pipeline, unload_lora_from_pipeline


class PipelineManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._pipeline: StableDiffusionPipeline | None = None

    def generate_image(
        self,
        *,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        denoise_strength: float,
        seed: int,
        lora_style: str,
        lora_strength: float,
    ) -> Image.Image:
        with self._lock:
            pipeline = self._load_pipeline()
            lora_path = self._get_lora_path(lora_style)
            original_unet_state = copy.deepcopy(pipeline.unet.state_dict())
            original_te_state = copy.deepcopy(pipeline.text_encoder.state_dict())
            lora_loaded = False
            try:
                pipeline = load_a1111_lora_into_pipeline(pipeline, lora_path, lora_strength)
                lora_loaded = True
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = pipeline(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                )
                image = result.images[0]
            finally:
                if lora_loaded:
                    unload_lora_from_pipeline(pipeline, original_unet_state, original_te_state)
                self._unload_pipeline()

        return image

    def _load_pipeline(self) -> StableDiffusionPipeline:
        if self._pipeline is not None:
            return self._pipeline

        self._validate_yaml()
        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=str(self.settings.model_path),
            original_config_file=str(self.settings.model_config_path),
            torch_dtype=torch.float32,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipeline.set_progress_bar_config(disable=False)
        pipeline.to("cpu")
        self._pipeline = pipeline
        return pipeline

    def _unload_pipeline(self) -> None:
        if self._pipeline is None:
            return

        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_yaml(self) -> None:
        with self.settings.model_config_path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] | None = yaml.safe_load(handle)

        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("Model config YAML is empty or invalid.")

    @staticmethod
    def _get_lora_path(lora_style: str) -> str:
        try:
            return LORA_PATHS[lora_style]
        except KeyError as exc:
            raise ValueError(f"Unsupported LoRA style: {lora_style}") from exc
