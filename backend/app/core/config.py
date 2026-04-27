from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


LORA_PATHS = {
    "anime": "/home/seechan1/Desktop/comics/models/lora/AnimeAnything_SD15.safetensors",
    "manga": "/home/seechan1/Desktop/comics/models/lora/animeoutlineV4_16.safetensors",
    "comic": "/home/seechan1/Desktop/comics/models/lora/Comic_book_V2.safetensors",
}


class Settings(BaseSettings):
    app_name: str = "Comics Local Generator API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    frontend_origin: str = "http://127.0.0.1:5173"
    model_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/basemodel/v1-5-pruned-emaonly.safetensors")
    )
    model_config_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/config_files/v1-inference.yaml")
    )
    controlnet_model_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/controlnet/control_v11p_sd15_lineart.safetensors")
    )
    controlnet_config_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/config_files/control_v11p_sd15_lineart.yaml")
    )
    controlnet_diffusers_config_dir: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/controlnet/control_v11p_sd15_lineart_diffusers")
    )
    controlnet_annotator_cache_dir: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/controlnet/annotators")
    )
    ip_adapter_dir: Path = Field(default=Path("/home/seechan1/Desktop/comics/models/ip-adapter"))
    ip_adapter_model_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/ip-adapter/ip-adapter-plus_sd15.bin")
    )
    ip_adapter_image_encoder_weights_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/ip-adapter/image_encoder.bin")
    )
    ip_adapter_image_encoder_config_path: Path = Field(
        default=Path("/home/seechan1/Desktop/comics/models/ip-adapter/image_encoder_config.json")
    )
    output_dir: Path = Field(default=Path("/home/seechan1/Desktop/comics/output"))
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 20
    default_cfg_scale: float = 7.5
    default_negative_prompt: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
