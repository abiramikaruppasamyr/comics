from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.inpaint_pipeline import InpaintService


router = APIRouter()
inpaint_service = InpaintService()


@router.post("/generate")
async def generate_inpaint_image(
    init_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    positive_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(20),
    cfg_scale: float = Form(7.5),
    denoise_strength: float = Form(1.0),
    seed: int = Form(-1),
    lora_style: str = Form(...),
    lora_strength: float = Form(1.0),
) -> dict[str, float | int | str]:
    init_temp_path: Path | None = None
    mask_temp_path: Path | None = None
    try:
        init_image_bytes = await init_image.read()
        mask_image_bytes = await mask_image.read()
        if not init_image_bytes:
            raise ValueError("Initial image upload is empty.")
        if not mask_image_bytes:
            raise ValueError("Mask image upload is empty.")

        with NamedTemporaryFile(delete=False, dir="/tmp", suffix=Path(init_image.filename or "init.png").suffix or ".png") as init_temp:
            init_temp.write(init_image_bytes)
            init_temp_path = Path(init_temp.name)

        with NamedTemporaryFile(delete=False, dir="/tmp", suffix=Path(mask_image.filename or "mask.png").suffix or ".png") as mask_temp:
            mask_temp.write(mask_image_bytes)
            mask_temp_path = Path(mask_temp.name)

        result = inpaint_service.inpaint(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            init_image_path=str(init_temp_path),
            mask_image_path=str(mask_temp_path),
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            denoise_strength=denoise_strength,
            seed=seed,
            lora_style=lora_style,
            lora_strength=lora_strength,
        )
        return {
            "image_url": f"/output/{result['image_filename']}",
            "cpu_usage": result["cpu_usage"],
            "ram_used": result["ram_used"],
            "ram_total": result["ram_total"],
            "seed_used": result["seed_used"],
            "generation_time_seconds": result["generation_time_seconds"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if init_temp_path is not None and init_temp_path.exists():
            init_temp_path.unlink(missing_ok=True)
        if mask_temp_path is not None and mask_temp_path.exists():
            mask_temp_path.unlink(missing_ok=True)
