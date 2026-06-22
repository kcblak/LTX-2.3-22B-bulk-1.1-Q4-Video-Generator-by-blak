from __future__ import annotations

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("LTX.Config")


class ConfigError(Exception):
    pass


REQUIRED_FIELDS = {
    "project_name": str,
    "mode": str,
    "device": str,
    "batch_size": int,
    "max_retries": int,
    "parallel_jobs": int,
    "output_format": str,
    "resolution_default": str,
    "logging_level": str,
}


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data
        self.project_name: str = data.get("project_name", "DefaultProject")
        self.mode: str = data.get("mode", "headless")
        self.device: str = data.get("device", "cuda")
        self.batch_size: int = int(data.get("batch_size", 1))
        self.max_retries: int = int(data.get("max_retries", 3))
        self.parallel_jobs: int = int(data.get("parallel_jobs", 1))
        self.output_format: str = data.get("output_format", "mp4")
        self.resolution_default: str = data.get("resolution_default", "720p")
        self.logging_level: str = data.get("logging_level", "info").lower()

        self.model_name: str = data.get("model_name", "ltx2_22B_distilled")
        self.fps: int = int(data.get("fps", 24))
        self.enhance_prompt: bool = bool(data.get("enhance_prompt", False))
        self.guide_scale: float = float(data.get("guide_scale", 3.0))
        self.sampling_steps: int = int(data.get("sampling_steps", 8))
        self.guide_phases: int = int(data.get("guide_phases", 2))
        self.frame_interpolation: bool = bool(data.get("frame_interpolation", False))
        self.upscale: bool = bool(data.get("upscale", False))
        self.auto_upload_drive: bool = bool(data.get("auto_upload_drive", False))
        self.drive_folder_id: Optional[str] = data.get("drive_folder_id") or None
        self.drive_shared_drive_id: Optional[str] = data.get("drive_shared_drive_id") or None
        self.service_account_json_path: Optional[str] = data.get("service_account_json_path") or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "mode": self.mode,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "parallel_jobs": self.parallel_jobs,
            "output_format": self.output_format,
            "resolution_default": self.resolution_default,
            "logging_level": self.logging_level,
            "model_name": self.model_name,
            "fps": self.fps,
            "enhance_prompt": self.enhance_prompt,
            "guide_scale": self.guide_scale,
            "sampling_steps": self.sampling_steps,
            "guide_phases": self.guide_phases,
            "frame_interpolation": self.frame_interpolation,
            "upscale": self.upscale,
            "auto_upload_drive": self.auto_upload_drive,
            "drive_folder_id": self.drive_folder_id,
            "drive_shared_drive_id": self.drive_shared_drive_id,
            "service_account_json_path": self.service_account_json_path,
        }


def _validate(data: dict[str, Any]) -> None:
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            raise ConfigError(f"Missing required config field: {field}")
        if not isinstance(data[field], expected_type):
            raise ConfigError(
                f"Field '{field}' must be {expected_type.__name__}, got {type(data[field]).__name__}"
            )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")
    _validate(data)
    LOGGER.info("Loaded config from %s", path)
    return Config(data)
