from __future__ import annotations

import gc
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import psutil
import torch
from controlnet_aux.lineart import LineartDetector
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline
from PIL import Image

from app.core.config import get_settings


UINT32_MAX = 2**32


class ControlNetLineartService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(
        self,
        *,
        image_bytes: bytes,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        controlnet_conditioning_scale: float = 1.0,
    ) -> dict[str, float | int | str]:
        self._validate_files()

        started_at = time.perf_counter()
        seed_used = self._resolve_seed(seed)
        lineart_image = self._preprocess_image(image_bytes=image_bytes, width=width, height=height)

        timestamp = datetime.now(timezone.utc)
        lineart_filename = self._build_filename(prefix="lineart", timestamp=timestamp)
        output_filename = self._build_filename(prefix="controlnet", timestamp=timestamp)

        lineart_path = self.settings.output_dir / lineart_filename
        output_path = self.settings.output_dir / output_filename
        lineart_image.save(lineart_path)

        pipeline = None
        controlnet = None
        try:
            controlnet = ControlNetModel.from_single_file(
                pretrained_model_link_or_path=str(self.settings.controlnet_model_path),
                config=str(self.settings.controlnet_diffusers_config_dir),
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            pipeline = StableDiffusionControlNetPipeline.from_single_file(
                pretrained_model_link_or_path=str(self.settings.model_path),
                original_config_file=str(self.settings.model_config_path),
                controlnet=controlnet,
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
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to("cpu")

            generator = torch.Generator(device="cpu").manual_seed(seed_used)
            result = pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=lineart_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            )
            generated_image = result.images[0]
            generated_image.save(output_path)
        finally:
            if pipeline is not None:
                del pipeline
            if controlnet is not None:
                del controlnet
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        generation_time_seconds = round(time.perf_counter() - started_at, 2)
        memory = psutil.virtual_memory()
        return {
            "image_url": f"/output/{output_filename}",
            "lineart_preview_url": f"/output/{lineart_filename}",
            "cpu_usage": round(psutil.cpu_percent(interval=0.2), 1),
            "ram_used": round(memory.used / (1024 * 1024), 1),
            "ram_total": round(memory.total / (1024 * 1024), 1),
            "seed_used": seed_used,
            "generation_time_seconds": generation_time_seconds,
            "image_filename": output_filename,
            "preprocessed_lineart_filename": lineart_filename,
        }

    def _preprocess_image(self, *, image_bytes: bytes, width: int, height: int) -> Image.Image:
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((width, height), Image.LANCZOS)
        annotator_dir = self.settings.controlnet_annotator_cache_dir / "lllyasviel" / "Annotators"
        required_annotators = [
            annotator_dir / "sk_model.pth",
            annotator_dir / "sk_model2.pth",
        ]
        missing_annotators = [str(path) for path in required_annotators if not path.exists()]
        if missing_annotators:
            raise RuntimeError(
                "Local ControlNet annotator weights are missing: "
                + ", ".join(missing_annotators)
            )

        try:
            detector = LineartDetector.from_pretrained(
                str(annotator_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load local ControlNet lineart annotator weights from "
                f"{annotator_dir}."
            ) from exc

        try:
            detector.to("cpu")
            lineart_image = detector(
                input_image,
                coarse=False,
                detect_resolution=max(width, height),
                image_resolution=max(width, height),
            )
        finally:
            del detector
            gc.collect()

        if not isinstance(lineart_image, Image.Image):
            lineart_image = Image.fromarray(lineart_image)

        return lineart_image.convert("RGB").resize((width, height), Image.LANCZOS)

    def _validate_files(self) -> None:
        required_paths: list[Path] = [
            self.settings.model_path,
            self.settings.model_config_path,
            self.settings.controlnet_model_path,
            self.settings.controlnet_config_path,
            self.settings.controlnet_diffusers_config_dir / "config.json",
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
