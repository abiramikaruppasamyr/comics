from __future__ import annotations

import copy
import gc
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import psutil
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
from PIL import Image

from app.core.config import LORA_PATHS, get_settings
from app.services.lora_loader import load_a1111_lora_into_pipeline, unload_lora_from_pipeline


UINT32_MAX = 2**32


class InpaintService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.inpainting_model_path = Path("/home/seechan1/Desktop/comics/models/inpainting/sd-v1-5-inpainting.safetensors")
        self.original_config_path = Path("/home/seechan1/Desktop/comics/models/config_files/v1-inference.yaml")

    def inpaint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        init_image_path: str,
        mask_image_path: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        denoise_strength: float,
        seed: int,
        lora_style: str,
        lora_strength: float,
    ) -> dict[str, float | int | str]:
        self._validate_files(init_image_path=init_image_path, mask_image_path=mask_image_path)

        started_at = time.perf_counter()
        seed_used = self._resolve_seed(seed)
        init_image = Image.open(init_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
        mask_image = Image.open(mask_image_path).convert("RGB").resize((width, height), Image.LANCZOS).convert("L")

        timestamp = datetime.now(timezone.utc)
        output_filename = self._build_filename(prefix="inpaint", timestamp=timestamp)
        output_path = self.settings.output_dir / output_filename

        pipeline = None
        original_unet_state = None
        original_te_state = None
        lora_loaded = False
        try:
            pipeline = StableDiffusionInpaintPipeline.from_single_file(
                pretrained_model_link_or_path=str(self.inpainting_model_path),
                original_config_file=str(self.original_config_path),
                num_in_channels=9,
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

            lora_path = self._get_lora_path(lora_style)
            original_unet_state = copy.deepcopy(pipeline.unet.state_dict())
            original_te_state = copy.deepcopy(pipeline.text_encoder.state_dict())
            pipeline = load_a1111_lora_into_pipeline(pipeline, lora_path, lora_strength)
            lora_loaded = True

            generator = torch.Generator(device="cpu").manual_seed(seed_used)
            result = pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                strength=denoise_strength,
                guidance_scale=cfg_scale,
                generator=generator,
            )
            generated_image = result.images[0]
            generated_image.save(output_path)
        finally:
            if pipeline is not None and lora_loaded:
                unload_lora_from_pipeline(pipeline, original_unet_state, original_te_state)
            if pipeline is not None:
                del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        generation_time_seconds = round(time.perf_counter() - started_at, 2)
        memory = psutil.virtual_memory()
        return {
            "image_filename": output_filename,
            "cpu_usage": round(psutil.cpu_percent(interval=0.2), 1),
            "ram_used": round(memory.used / (1024 * 1024), 1),
            "ram_total": round(memory.total / (1024 * 1024), 1),
            "seed_used": seed_used,
            "generation_time_seconds": generation_time_seconds,
        }

    def _validate_files(self, *, init_image_path: str, mask_image_path: str) -> None:
        required_paths = [
            self.inpainting_model_path,
            self.original_config_path,
            Path(init_image_path),
            Path(mask_image_path),
            *(Path(path) for path in LORA_PATHS.values()),
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing required model files: {', '.join(missing_paths)}")

    @staticmethod
    def _build_filename(*, prefix: str, timestamp: datetime) -> str:
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{stamp}_{uuid4().hex[:8]}.png"

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        if seed == -1:
            return int(torch.randint(0, UINT32_MAX, (1,), device="cpu").item())
        return seed % UINT32_MAX

    @staticmethod
    def _get_lora_path(lora_style: str) -> str:
        try:
            return LORA_PATHS[lora_style]
        except KeyError as exc:
            raise ValueError(f"Unsupported LoRA style: {lora_style}") from exc
