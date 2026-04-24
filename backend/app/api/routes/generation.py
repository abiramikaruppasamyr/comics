from fastapi import APIRouter, HTTPException

from app.schemas.generation import GenerateImageRequest, GenerateImageResponse
from app.services.generator import ImageGeneratorService


router = APIRouter()
generator_service = ImageGeneratorService()


@router.post("", response_model=GenerateImageResponse)
def generate_image(payload: GenerateImageRequest) -> GenerateImageResponse:
    try:
        return generator_service.generate(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
