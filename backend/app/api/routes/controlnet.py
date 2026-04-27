from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.controlnet_pipeline import ControlNetLineartService


router = APIRouter()
controlnet_service = ControlNetLineartService()


@router.post("/generate")
async def generate_controlnet_image(
    image: UploadFile = File(...),
    positive_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    steps: int = Form(20),
    cfg_scale: float = Form(7.5),
    denoise_strength: float = Form(1.0),
    seed: int = Form(-1),
    controlnet_conditioning_scale: float = Form(1.0),
    lora_style: str = Form(...),
    lora_strength: float = Form(1.0),
) -> dict[str, float | int | str]:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        return controlnet_service.generate(
            image_bytes=image_bytes,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            denoise_strength=denoise_strength,
            seed=seed,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            lora_style=lora_style,
            lora_strength=lora_strength,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
