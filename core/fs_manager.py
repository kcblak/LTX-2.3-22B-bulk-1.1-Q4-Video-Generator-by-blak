from __future__ import annotations

import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from core.logger import StructuredFormatter
from core.config import Config


class FSManager:
    def __init__(self, config: Config):
        self.config = config

        # Kaggle or local root
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            self.drive_root = Path("/kaggle/working/MyDrive")
        else:
            self.drive_root = Path("MyDrive")

        self.project_root = self.drive_root / "LTX_PROJECTS" / config.project_name
        self.input_dir = self.project_root / "input"
        self.output_dir = self.project_root / "output"
        self.logs_dir = self.project_root / "logs"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.cache_dir = self.project_root / "cache"
        self.config_dir = self.project_root / "config"

        self.input_images_dir = self.input_dir / "images"
        self.input_zips_dir = self.input_dir / "zips"
        self.input_extracted_dir = self.input_dir / "extracted"

        self.output_videos_dir = self.output_dir / "videos"
        self.output_frames_dir = self.output_dir / "frames"
        self.output_thumbnails_dir = self.output_dir / "thumbnails"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        dirs = [
            self.project_root,
            self.input_dir,
            self.output_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.cache_dir,
            self.config_dir,
            self.input_images_dir,
            self.input_zips_dir,
            self.input_extracted_dir,
            self.output_videos_dir,
            self.output_frames_dir,
            self.output_thumbnails_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def project_config_path(self) -> Path:
        return self.config_dir / "config.yaml"

    def jobs_csv_path(self) -> Path:
        return self.input_dir / "jobs.csv"

    def validation_report_path(self) -> Path:
        return self.logs_dir / "validation_report.txt"

    def status_csv_path(self) -> Path:
        return self.checkpoints_dir / "job_status.csv"

    def checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "pipeline_state.json"

    def cell_checkpoint_path(self, cell_name: str) -> Path:
        return self.checkpoints_dir / f"cell_{cell_name}.done"

    def job_output_path(self, job_id: str, ext: str = "mp4") -> Path:
        return self.output_videos_dir / f"job_{job_id}.{ext}"

    def job_frames_dir(self, job_id: str) -> Path:
        d = self.output_frames_dir / f"job_{job_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def job_thumbnail_path(self, job_id: str) -> Path:
        return self.output_thumbnails_dir / f"job_{job_id}.jpg"

    def job_log_path(self, job_id: str) -> Path:
        return self.logs_dir / f"job_{job_id}.log"

    def extract_zips(self) -> list[Path]:
        extracted: list[Path] = []
        for zip_path in self.input_zips_dir.glob("*.zip"):
            target = self.input_extracted_dir / zip_path.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.infolist():
                    member_path = PurePosixPath(member.filename)
                    if any(part in {"", ".", ".."} for part in member_path.parts):
                        continue
                    out_path = (target / member_path).resolve()
                    if not str(out_path).startswith(str(target.resolve())):
                        continue
                    zf.extract(member, target)
                    extracted.append(out_path)
        return extracted

    def write_project_config(self) -> None:
        import yaml
        sample = {
            "project_name": self.config.project_name,
            "mode": "headless",
            "device": "cuda",
            "batch_size": 1,
            "max_retries": 3,
            "parallel_jobs": 1,
            "output_format": "mp4",
            "resolution_default": "720p",
            "logging_level": "info",
            "fps": 24,
            "enhance_prompt": False,
            "guide_scale": 3.0,
            "sampling_steps": 8,
            "guide_phases": 2,
            "frame_interpolation": False,
            "upscale": False,
            "auto_upload_drive": False,
            "drive_folder_id": "",
            "drive_shared_drive_id": "",
            "service_account_json_path": "",
        }
        with self.project_config_path().open("w", encoding="utf-8") as f:
            yaml.dump(sample, f, default_flow_style=False, sort_keys=False)

    def summary(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "jobs_csv": str(self.jobs_csv_path()),
            "status_csv": str(self.status_csv_path()),
            "logs_dir": str(self.logs_dir),
            "output_videos": str(self.output_videos_dir),
        }
