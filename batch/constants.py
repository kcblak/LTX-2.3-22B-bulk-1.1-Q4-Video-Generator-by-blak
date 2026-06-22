from __future__ import annotations

import os
from pathlib import Path

REQUIRED_COLUMNS = [
    "prompt",
    "start_image",
    "end_image",
    "duration",
    "resolution",
    "aspect_ratio",
    "seed",
    "guide_scale",
    "steps",
]

DURATION_CHOICES = {
    "2 Seconds (49 frames)",
    "3 Seconds (73 frames)",
    "5 Seconds (121 frames)",
    "8 Seconds (193 frames)",
    "10 Seconds (241 frames)",
    "15 Seconds (361 frames)",
}

RESOLUTION_CHOICES = {"1080p", "720p", "540p", "480p"}
ASPECT_RATIO_CHOICES = {"16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def default_kaggle_root() -> Path:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working")
    return Path.cwd()
