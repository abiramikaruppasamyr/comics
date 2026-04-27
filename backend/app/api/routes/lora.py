from fastapi import APIRouter


router = APIRouter()


@router.get("/styles")
def list_lora_styles() -> list[dict[str, float | str]]:
    return [
        {"key": "anime", "label": "Anime", "default_strength": 1.0},
        {"key": "manga", "label": "Manga / Lineart", "default_strength": 1.0},
        {"key": "comic", "label": "Comic Book", "default_strength": 1.0},
    ]
