from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile


router = APIRouter()
ip_adapter_service = None


def get_ip_adapter_service():
    global ip_adapter_service
    if ip_adapter_service is None:
        from app.services.ip_adapter_pipeline import IPAdapterService

        ip_adapter_service = IPAdapterService()
    return ip_adapter_service


@router.post("/generate")
async def generate_ip_adapter_image(
    reference_images: list[UploadFile] = File(...),
    positive_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    ip_adapter_scale: float = Form(0.6),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(20),
    cfg_scale: float = Form(7.5),
    denoise_strength: float = Form(1.0),
    seed: int = Form(-1),
    lora_style: str = Form(...),
    lora_strength: float = Form(1.0),
) -> dict[str, float | int | str]:
    temp_paths: list[Path] = []
    try:
        if len(reference_images) == 0:
            raise ValueError("Upload at least one reference image.")

        for upload in reference_images:
            image_bytes = await upload.read()
            if not image_bytes:
                raise ValueError(f"Reference image upload is empty: {upload.filename or 'unnamed file'}")

            suffix = Path(upload.filename or "reference.png").suffix or ".png"
            with NamedTemporaryFile(delete=False, dir="/tmp", suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                temp_paths.append(Path(temp_file.name))

        result = get_ip_adapter_service().generate(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_image_paths=[str(path) for path in temp_paths],
            ip_adapter_scale=ip_adapter_scale,
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
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
