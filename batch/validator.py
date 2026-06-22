from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ASPECT_RATIO_CHOICES,
    DURATION_CHOICES,
    IMAGE_EXTENSIONS,
    REQUIRED_COLUMNS,
    RESOLUTION_CHOICES,
)


class CSVValidationError(ValueError):
    pass


@dataclass
class ValidationResult:
    passed: bool
    jobs: list[dict[str, Any]]
    errors: list[str]
    report_path: Path | None = None

    def write_report(self, path: Path, manifest_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "LTX Batch Validation Report",
            "===========================",
            f"Manifest: {manifest_path}",
            f"Total Rows: {len(self.jobs)}",
            f"Status: {'PASS' if self.passed else 'FAIL'}",
            "",
        ]
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"- {error}")
        else:
            lines.append("Errors: none")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_header(header: str | None) -> str:
    return (header or "").strip()


def read_jobs(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise CSVValidationError(f"CSV manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVValidationError("CSV manifest is empty or has no header row.")

        normalized = [_strip_header(column) for column in reader.fieldnames]
        if normalized != REQUIRED_COLUMNS:
            expected = ",".join(REQUIRED_COLUMNS)
            actual = ",".join(normalized)
            raise CSVValidationError(
                f"CSV header must exactly match: {expected}. Actual header: {actual}"
            )

        jobs: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                cleaned[key.strip()] = value.strip() if isinstance(value, str) else value

            prompt = cleaned.get("prompt", "")
            if prompt == "":
                raise CSVValidationError(f"Row {line_number}: prompt is required and cannot be empty.")

            job = {
                "row": line_number - 1,
                "prompt": prompt,
                "start_image": cleaned.get("start_image", ""),
                "end_image": cleaned.get("end_image", ""),
                "duration": cleaned.get("duration", ""),
                "resolution": cleaned.get("resolution", ""),
                "aspect_ratio": cleaned.get("aspect_ratio", ""),
                "seed": cleaned.get("seed", ""),
                "guide_scale": cleaned.get("guide_scale", ""),
                "steps": cleaned.get("steps", ""),
            }
            jobs.append(job)

    if not jobs:
        raise CSVValidationError("CSV manifest contains no data rows.")

    return jobs


def validate_job_values(jobs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for job in jobs:
        row = job["row"]
        duration = job.get("duration", "")
        resolution = job.get("resolution", "")
        aspect_ratio = job.get("aspect_ratio", "")

        if duration and duration not in DURATION_CHOICES:
            errors.append(f"Row {row}: invalid duration '{duration}'.")
        if resolution and resolution not in RESOLUTION_CHOICES:
            errors.append(f"Row {row}: invalid resolution '{resolution}'.")
        if aspect_ratio and aspect_ratio not in ASPECT_RATIO_CHOICES:
            errors.append(f"Row {row}: invalid aspect_ratio '{aspect_ratio}'.")

        seed_raw = job.get("seed", "")
        if seed_raw:
            try:
                int(seed_raw)
            except ValueError:
                errors.append(f"Row {row}: seed must be an integer or empty, got '{seed_raw}'.")

        guide_raw = job.get("guide_scale", "")
        if guide_raw:
            try:
                float(guide_raw)
            except ValueError:
                errors.append(f"Row {row}: guide_scale must be numeric or empty, got '{guide_raw}'.")

        steps_raw = job.get("steps", "")
        if steps_raw:
            try:
                int(steps_raw)
            except ValueError:
                errors.append(f"Row {row}: steps must be an integer or empty, got '{steps_raw}'.")

    return errors


def validate_and_report(
    manifest_path: str | Path,
    report_path: str | Path,
) -> ValidationResult:
    manifest = Path(manifest_path)
    errors: list[str] = []
    jobs: list[dict[str, Any]] = []

    try:
        jobs = read_jobs(manifest)
    except CSVValidationError as exc:
        errors.append(str(exc))

    if not errors:
        errors.extend(validate_job_values(jobs))

    result = ValidationResult(
        passed=not errors,
        jobs=jobs,
        errors=errors,
        report_path=Path(report_path),
    )
    result.write_report(Path(report_path), manifest)
    return result
