from __future__ import annotations

import secrets


UINT32_MAX = 2**32


def resolve_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbelow(UINT32_MAX)
    return seed % UINT32_MAX
