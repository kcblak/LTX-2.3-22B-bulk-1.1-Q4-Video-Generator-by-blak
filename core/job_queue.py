from __future__ import annotations

import glob
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from batch.validator import read_jobs, validate_job_values, ValidationResult
from batch.image_resolver import ImageResolver, validate_images_and_report
from batch.state_manager import StateManager
from batch.status import StatusWriter
from core.fs_manager import FSManager
from core.logger import Logger
from core.config import Config


class JobQueue:
    def __init__(self, fs: FSManager, logger: Logger):
        self.fs = fs
        self.logger = logger
        self.state = StateManager(fs.checkpoints_dir)
        self.status = StatusWriter(fs.status_csv_path)
        self.jobs: list[dict] = []
        self._resolver: Optional[ImageResolver] = None

    def load(self) -> ValidationResult:
        csv_path = self.fs.jobs_csv_path()
        report_path = self.fs.validation_report_path()
        result = validate_images_and_report(
            jobs=read_jobs(csv_path),
            manifest_path=csv_path,
            images_dir=self.fs.input_images_dir,
            images_zip=self._find_zip(),
            validation=validate_and_report_internal(csv_path, report_path),
        )
        self.jobs = result.jobs
        self.status.initialize(self.jobs)
        self.state.save(csv_path)
        return result

    def _find_zip(self) -> Optional[Path]:
        zips = sorted(self.fs.input_zips_dir.glob("*.zip"))
        return zips[0] if zips else None

    def pending_jobs(self) -> list[dict]:
        last_row = self.state.load().get("last_completed_row")
        pending = []
        for job in self.jobs:
            row = int(job["row"])
            status_data = self.status.rows.get(row, {})
            status = status_data.get("status", "pending")
            if status in ("pending", "failed") and (last_row is None or row > last_row):
                pending.append(job)
        return pending

    def mark_running(self, job: dict) -> None:
        self.status.update(
            row=int(job["row"]),
            status="running",
            start_time=_now_iso(),
        )

    def mark_completed(self, job: dict, output_file: str) -> None:
        self.status.update(
            row=int(job["row"]),
            status="completed",
            output_file=output_file,
            end_time=_now_iso(),
            duration=_elapsed_seconds(
                self.status.rows.get(int(job["row"]), {}).get("start_time", "")
            ),
        )
        self.state.mark_completed(int(job["row"]))

    def mark_failed(self, job: dict, error: str) -> None:
        self.status.update(
            row=int(job["row"]),
            status="failed",
            error=str(error),
            end_time=_now_iso(),
        )


def validate_and_report_internal(
    manifest_path: Path, report_path: Path
) -> ValidationResult:
    from batch.validator import validate_and_report
    return validate_and_report(manifest_path, report_path)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(start_iso: str) -> str:
    if not start_iso:
        return ""
    try:
        from datetime import datetime, timezone
        start = datetime.fromisoformat(start_iso)
        end = datetime.now(timezone.utc)
        return f"{(end - start).total_seconds():.2f}"
    except Exception:
        return ""
