from __future__ import annotations

import gc
import math
import threading
import time
from pathlib import Path

import psutil
import torch
from controlnet_aux.lineart import LineartDetector
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetInpaintPipeline
from PIL import Image, ImageOps

from app.core.config import get_settings


UINT32_MAX = 2**32


class InpaintService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()

    def generate(
        self,
        *,
        base_image: Image.Image,
        mask_image: Image.Image,
        control_image: Image.Image | None,
        prompt: str,
        negative_prompt: str,
        target_width: int | None,
        target_height: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
    ) -> Image.Image:
        self._validate_inputs(
            prompt=prompt,
            target_width=target_width,
            target_height=target_height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        self._validate_files()

        seed_used = self.resolve_seed(seed)
        scheduler_step_count = self.get_scheduler_step_count(
            requested_steps=num_inference_steps,
            strength=strength,
        )
        started_at = time.perf_counter()
        output_width = target_width or base_image.width
        output_height = target_height or base_image.height
        prepared_base = self._fit_with_white_padding(ImageOps.exif_transpose(base_image).convert("RGB"), 512, 512)
        original_base = prepared_base.copy()
        prepared_mask = self._fit_with_black_padding(ImageOps.exif_transpose(mask_image).convert("L"), 512, 512)
        prepared_mask = prepared_mask.point(lambda pixel: 255 if pixel > 127 else 0, mode="L")
        has_control_reference = control_image is not None
        control_source = (
            prepared_base
            if not has_control_reference
            else ImageOps.exif_transpose(control_image).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        )
        generation_base = (
            self._clear_masked_area(prepared_base, prepared_mask)
            if has_control_reference
            else prepared_base
        )
        prepared_control = self._preprocess_control_image(control_source, width=512, height=512)
        effective_strength = 1.0 if has_control_reference else strength
        controlnet_conditioning_scale = 2.0 if has_control_reference else 1.0

        with self._lock:
            pipeline = None
            controlnet = None
            try:
                try:
                    controlnet = ControlNetModel.from_single_file(
                        pretrained_model_link_or_path=str(self.settings.controlnet_model_path),
                        config=str(self.settings.controlnet_diffusers_config_dir),
                        torch_dtype=torch.float32,
                        local_files_only=True,
                    )
                    pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
                        pretrained_model_link_or_path=str(self.settings.inpaint_model_path),
                        original_config_file=str(self.settings.model_config_path),
                        controlnet=controlnet,
                        torch_dtype=torch.float32,
                        local_files_only=True,
                        safety_checker=None,
                        requires_safety_checker=False,
                        num_in_channels=9,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        "Required inpaint or ControlNet model file was not found. "
                        f"Inpaint: {self.settings.inpaint_model_path}; "
                        f"ControlNet: {self.settings.controlnet_model_path}"
                    ) from exc
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=True,
                )
                pipeline.set_progress_bar_config(disable=False)
                pipeline.to("cpu")

                generator = torch.Generator(device="cpu").manual_seed(seed_used)
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=generation_base,
                    mask_image=prepared_mask,
                    control_image=prepared_control,
                    width=512,
                    height=512,
                    num_inference_steps=scheduler_step_count,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    strength=effective_strength,
                    generator=generator,
                )
                output_image = result.images[0].convert("RGB")
                final_image = Image.composite(output_image, original_base, prepared_mask)
                if final_image.size != (output_width, output_height):
                    final_image = final_image.resize((output_width, output_height), Image.Resampling.LANCZOS)
                memory = psutil.virtual_memory()
                print(
                    "[inpaint] "
                    f"generation_time_seconds={round(time.perf_counter() - started_at, 2)} "
                    f"seed_used={seed_used} "
                    f"steps={num_inference_steps} "
                    f"pipeline_steps={scheduler_step_count} "
                    f"control_reference={has_control_reference} "
                    f"controlnet_conditioning_scale={controlnet_conditioning_scale} "
                    f"effective_strength={effective_strength} "
                    f"ram_used_mb={round(memory.used / (1024 * 1024), 1)} "
                    f"ram_total_mb={round(memory.total / (1024 * 1024), 1)}"
                )
                return final_image
            finally:
                if pipeline is not None:
                    del pipeline
                if controlnet is not None:
                    del controlnet
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _validate_files(self) -> None:
        annotator_dir = self.settings.controlnet_annotator_cache_dir / "lllyasviel" / "Annotators"
        required_paths: list[Path] = [
            self.settings.inpaint_model_path,
            self.settings.model_config_path,
            self.settings.controlnet_model_path,
            self.settings.controlnet_config_path,
            self.settings.controlnet_diffusers_config_dir / "config.json",
            annotator_dir / "sk_model.pth",
            annotator_dir / "sk_model2.pth",
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing required inpaint ControlNet model files: {', '.join(missing_paths)}")

    def _preprocess_control_image(self, image: Image.Image, *, width: int, height: int) -> Image.Image:
        input_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        annotator_dir = self.settings.controlnet_annotator_cache_dir / "lllyasviel" / "Annotators"
        try:
            detector = LineartDetector.from_pretrained(str(annotator_dir))
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

        return lineart_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _clear_masked_area(image: Image.Image, mask: Image.Image) -> Image.Image:
        neutral_canvas = Image.new("RGB", image.size, "white")
        return Image.composite(neutral_canvas, image, mask)

    @staticmethod
    def _validate_inputs(
        *,
        prompt: str,
        target_width: int | None,
        target_height: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
    ) -> None:
        if not prompt.strip():
            raise ValueError("Prompt is required.")
        if target_width is not None and target_width < 64:
            raise ValueError("Width must be at least 64.")
        if target_height is not None and target_height < 64:
            raise ValueError("Height must be at least 64.")
        if num_inference_steps < 1:
            raise ValueError("Steps must be at least 1.")
        if guidance_scale <= 0:
            raise ValueError("Guidance scale must be greater than 0.")
        if strength <= 0 or strength > 1:
            raise ValueError("Strength must be greater than 0 and less than or equal to 1.")

    @staticmethod
    def resolve_seed(seed: int) -> int:
        if seed == -1:
            return int(torch.randint(0, UINT32_MAX, (1,), device="cpu").item())
        return seed % UINT32_MAX

    @staticmethod
    def get_scheduler_step_count(*, requested_steps: int, strength: float) -> int:
        # Diffusers inpaint internally uses strength * num_inference_steps denoising steps.
        return max(requested_steps, math.ceil(requested_steps / strength))

    @staticmethod
    def _nearest_multiple_of_8(value: int) -> int:
        return max(8, int(round(value / 8)) * 8)

    @staticmethod
    def _fit_with_white_padding(image: Image.Image, width: int, height: int) -> Image.Image:
        canvas = Image.new("RGB", (width, height), "white")
        scale = min(width / image.width, height / image.height)
        resized_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized_image = image.resize(resized_size, Image.Resampling.LANCZOS)
        offset = ((width - resized_size[0]) // 2, (height - resized_size[1]) // 2)
        canvas.paste(resized_image, offset)
        return canvas

    @staticmethod
    def _fit_with_black_padding(image: Image.Image, width: int, height: int) -> Image.Image:
        canvas = Image.new("L", (width, height), 0)
        scale = min(width / image.width, height / image.height)
        resized_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized_image = image.resize(resized_size, Image.Resampling.LANCZOS)
        offset = ((width - resized_size[0]) // 2, (height - resized_size[1]) // 2)
        canvas.paste(resized_image, offset)
        return canvas
