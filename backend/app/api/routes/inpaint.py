from __future__ import annotations

import base64
import time
from io import BytesIO
from uuid import uuid4

import psutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import get_settings


router = APIRouter()
inpaint_service = None


def get_inpaint_service():
    global inpaint_service
    if inpaint_service is None:
        from app.services.inpaint_pipeline import InpaintService

        inpaint_service = InpaintService()
    return inpaint_service


@router.post("")
async def generate_inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    control_image: UploadFile | None = File(None),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.3),
    seed: int = Form(-1),
) -> dict[str, float | int | str]:
    started_at = time.perf_counter()
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        if not mask_bytes:
            raise ValueError("Uploaded mask is empty.")
        control_image_bytes = await control_image.read() if control_image is not None else None

        base_image = _decode_image(image_bytes, mode="RGB")
        mask_image = _decode_image(mask_bytes, mode="L")
        control_reference_image = _decode_image(control_image_bytes, mode="RGB") if control_image_bytes else None
        service = get_inpaint_service()
        seed_used = service.resolve_seed(seed)
        pipeline_steps = service.get_scheduler_step_count(
            requested_steps=steps,
            strength=strength,
        )
        result_image = service.generate(
            base_image=base_image,
            mask_image=mask_image,
            control_image=control_reference_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            target_width=width if width > 0 else None,
            target_height=height if height > 0 else None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed_used,
        )

        output_dir = get_settings().output_dir / "inpainting"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_filename = f"inpaint_{time.strftime('%Y%m%d_%H%M%S')}_{seed_used}_{uuid4().hex[:8]}.png"
        output_path = output_dir / image_filename
        result_image.save(output_path, format="PNG")

        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        memory = psutil.virtual_memory()
        return {
            "image_base64": image_base64,
            "cpu_usage": round(psutil.cpu_percent(interval=0.2), 1),
            "ram_used": round(memory.used / (1024 * 1024), 1),
            "ram_total": round(memory.total / (1024 * 1024), 1),
            "seed_used": seed_used,
            "steps_used": steps,
            "pipeline_steps": pipeline_steps,
            "generation_time_seconds": round(time.perf_counter() - started_at, 2),
            "image_filename": image_filename,
            "image_url": f"/output/inpainting/{image_filename}",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _decode_image(image_bytes: bytes, *, mode: str) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    return ImageOps.exif_transpose(image).convert(mode)
