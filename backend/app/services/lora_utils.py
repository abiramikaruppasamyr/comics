from __future__ import annotations

import importlib.util
from pathlib import Path

from safetensors.torch import load_file


def ensure_peft_available() -> None:
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError(
            "PEFT backend is required for LoRA loading. Install `peft==0.14.0` in the backend environment."
        )


def split_lora_path(lora_path: str) -> tuple[Path, str]:
    path = Path(lora_path)
    return path.parent, path.name


def load_lora_checkpoint(lora_path: str) -> dict[str, object]:
    path = Path(lora_path)
    state_dict = load_file(str(path))

    # Diffusers 0.33.x does not apply `.diff_b` weights cleanly for these local checkpoints.
    # Filtering them out matches the library's own fallback path for unsupported entries.
    cleaned_state_dict = {
        key: value
        for key, value in state_dict.items()
        if ".diff_b" not in key
    }
    return cleaned_state_dict
