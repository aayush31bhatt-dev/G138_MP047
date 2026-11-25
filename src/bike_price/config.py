"""Configuration and file hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from YAML using safe_load."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
