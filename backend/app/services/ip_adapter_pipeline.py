from __future__ import annotations

import copy
import gc
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import psutil
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from app.core.config import LORA_PATHS, get_settings
from app.services.lora_loader import load_a1111_lora_into_pipeline, unload_lora_from_pipeline
from ip_adapter import IPAdapterPlus
from ip_adapter.ip_adapter import AttnProcessor, CNAttnProcessor, IPAttnProcessor, Resampler


UINT32_MAX = 2**32


class CPUIPAdapterPlus(IPAdapterPlus):
    def __init__(self, sd_pipe, image_encoder_path: str, ip_ckpt: str, device: str, num_tokens: int = 4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_path,
            local_files_only=True,
        ).to(self.device, dtype=torch.float32)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()
        self.image_proj_model.to(self.device, dtype=torch.float32)

    def init_proj(self):
        return Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float32)

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float32)

        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]

            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float32)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            if clip_image_embeds.shape[0] > 1:
                clip_image_embeds = clip_image_embeds.mean(dim=0, keepdim=True)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float32)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image = torch.zeros(
            (1, 3, 224, 224),
            device=self.device,
            dtype=torch.float32,
        )
        uncond_clip_image_embeds = self.image_encoder(
            uncond_clip_image,
            output_hidden_states=True,
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        negative_prompt_list = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt]

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image,
            clip_image_embeds=clip_image_embeds,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt_list,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt_list,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images


class IPAdapterService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        reference_image_paths: list[str],
        ip_adapter_scale: float,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        denoise_strength: float,
        seed: int,
        lora_style: str,
        lora_strength: float,
    ) -> dict[str, float | int | str]:
        image_encoder_folder = self._prepare_image_encoder_folder()
        self._validate_files(reference_image_paths=reference_image_paths, image_encoder_folder=image_encoder_folder)

        started_at = time.perf_counter()
        seed_used = self._resolve_seed(seed)
        torch.manual_seed(seed_used)
        timestamp = datetime.now(timezone.utc)
        output_filename = self._build_filename(prefix="ip_adapter", timestamp=timestamp)
        output_path = self.settings.output_dir / output_filename

        reference_images: list[Image.Image] = []
        pipeline = None
        ip_model = None
        original_unet_state = None
        original_te_state = None
        original_attn_processors = None
        lora_loaded = False
        try:
            for path in reference_image_paths:
                reference_images.append(Image.open(path).convert("RGB"))

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

            lora_path = self._get_lora_path(lora_style)
            original_unet_state = copy.deepcopy(pipeline.unet.state_dict())
            original_te_state = copy.deepcopy(pipeline.text_encoder.state_dict())
            pipeline = load_a1111_lora_into_pipeline(pipeline, lora_path, lora_strength)
            lora_loaded = True
            original_attn_processors = pipeline.unet.attn_processors.copy()

            ip_model = CPUIPAdapterPlus(
                pipeline,
                image_encoder_path=str(image_encoder_folder),
                ip_ckpt=str(self.settings.ip_adapter_model_path),
                device="cpu",
                num_tokens=16,
            )
            ip_model.image_encoder = ip_model.image_encoder.to(dtype=torch.float32)
            ip_model.image_proj_model = ip_model.image_proj_model.to(dtype=torch.float32)
            for _, module in ip_model.pipe.unet.named_modules():
                if hasattr(module, "weight") and module.weight is not None and module.weight.dtype == torch.float16:
                    module.to(torch.float32)

            ip_reference_image = self._prepare_ip_reference_image(reference_images)
            images = ip_model.generate(
                pil_image=ip_reference_image,
                num_samples=1,
                num_inference_steps=steps,
                seed=seed_used,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                scale=ip_adapter_scale,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
            )
            images[0].save(output_path)
        finally:
            for image in reference_images:
                image.close()
            if pipeline is not None and original_attn_processors is not None:
                pipeline.unet.set_attn_processor(original_attn_processors)
            if pipeline is not None and lora_loaded:
                unload_lora_from_pipeline(pipeline, original_unet_state, original_te_state)
            if ip_model is not None:
                del ip_model
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

    def _prepare_image_encoder_folder(self) -> Path:
        self.settings.ip_adapter_dir.mkdir(parents=True, exist_ok=True)
        required_links = {
            self.settings.ip_adapter_dir / "pytorch_model.bin": self.settings.ip_adapter_image_encoder_weights_path,
            self.settings.ip_adapter_dir / "config.json": self.settings.ip_adapter_image_encoder_config_path,
        }
        for target_path, source_path in required_links.items():
            if target_path.exists():
                continue
            self._link_or_copy(source_path=source_path, target_path=target_path)
        return self.settings.ip_adapter_dir

    @staticmethod
    def _link_or_copy(*, source_path: Path, target_path: Path) -> None:
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)

    def _prepare_ip_reference_image(self, reference_images: list[Image.Image]) -> Image.Image | list[Image.Image]:
        if len(reference_images) <= 1:
            return reference_images

        canvas_size = 512
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (245, 245, 245, 255))
        if len(reference_images) >= 4:
            ordered_images = [reference_images[-1], reference_images[0], reference_images[1], reference_images[2]]
        else:
            ordered_images = reference_images
        placements = self._reference_part_placements(len(ordered_images), canvas_size)

        for image, placement in zip(ordered_images, placements):
            part = self._crop_reference_content(image)
            if part.width == 0 or part.height == 0:
                continue

            max_width = placement["width"]
            max_height = placement["height"]
            scale = min(max_width / part.width, max_height / part.height)
            resized = part.resize(
                (max(1, int(part.width * scale)), max(1, int(part.height * scale))),
                Image.Resampling.LANCZOS,
            )
            x = int(placement["x"] - resized.width / 2)
            y = int(placement["y"] - resized.height / 2)
            canvas.alpha_composite(resized, dest=(x, y))

        return canvas.convert("RGB")

    @staticmethod
    def _reference_part_placements(count: int, canvas_size: int) -> list[dict[str, float]]:
        if count == 2:
            return [
                {"x": canvas_size * 0.5, "y": canvas_size * 0.38, "width": canvas_size * 0.8, "height": canvas_size * 0.35},
                {"x": canvas_size * 0.5, "y": canvas_size * 0.64, "width": canvas_size * 0.8, "height": canvas_size * 0.35},
            ]

        if count == 3:
            return [
                {"x": canvas_size * 0.5, "y": canvas_size * 0.32, "width": canvas_size * 0.8, "height": canvas_size * 0.25},
                {"x": canvas_size * 0.5, "y": canvas_size * 0.66, "width": canvas_size * 0.55, "height": canvas_size * 0.22},
                {"x": canvas_size * 0.5, "y": canvas_size * 0.49, "width": canvas_size * 0.35, "height": canvas_size * 0.28},
            ]

        placements = [
            {"x": canvas_size * 0.5, "y": canvas_size * 0.5, "width": canvas_size * 0.78, "height": canvas_size * 0.88},
            {"x": canvas_size * 0.5, "y": canvas_size * 0.36, "width": canvas_size * 0.58, "height": canvas_size * 0.18},
            {"x": canvas_size * 0.5, "y": canvas_size * 0.67, "width": canvas_size * 0.42, "height": canvas_size * 0.16},
            {"x": canvas_size * 0.5, "y": canvas_size * 0.51, "width": canvas_size * 0.28, "height": canvas_size * 0.22},
        ]
        return placements[:count]

    @staticmethod
    def _crop_reference_content(image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        alpha_bbox = rgba.getchannel("A").getbbox()
        if alpha_bbox is not None and alpha_bbox != (0, 0, rgba.width, rgba.height):
            return rgba.crop(alpha_bbox)

        rgb = np.asarray(rgba.convert("RGB"), dtype=np.int16)
        border_pixels = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
        background = np.median(border_pixels, axis=0)
        diff = np.abs(rgb - background).max(axis=2)
        mask = diff > 24
        if not mask.any():
            return rgba

        y_coords, x_coords = np.where(mask)
        padding = 8
        left = max(int(x_coords.min()) - padding, 0)
        upper = max(int(y_coords.min()) - padding, 0)
        right = min(int(x_coords.max()) + padding + 1, rgba.width)
        lower = min(int(y_coords.max()) + padding + 1, rgba.height)
        cropped = rgba.crop((left, upper, right, lower))

        cropped_rgb = np.asarray(cropped.convert("RGB"), dtype=np.int16)
        cropped_diff = np.abs(cropped_rgb - background).max(axis=2)
        alpha = np.where(cropped_diff > 24, 255, 0).astype(np.uint8)
        cropped.putalpha(Image.fromarray(alpha, mode="L"))
        return cropped

    def _validate_files(self, *, reference_image_paths: list[str], image_encoder_folder: Path) -> None:
        required_paths = [
            self.settings.model_path,
            self.settings.model_config_path,
            self.settings.ip_adapter_model_path,
            self.settings.ip_adapter_image_encoder_weights_path,
            self.settings.ip_adapter_image_encoder_config_path,
            image_encoder_folder / "pytorch_model.bin",
            image_encoder_folder / "config.json",
            *(Path(path) for path in reference_image_paths),
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
