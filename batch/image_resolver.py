from __future__ import annotations

import os
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from .constants import IMAGE_EXTENSIONS
from .validator import ValidationResult


class ImageResolutionError(ValueError):
    pass


class ImageResolver:
    def __init__(
        self,
        manifest_path: str | Path,
        images_dir: str | Path = "images",
        images_zip: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_parent = self.manifest_path.parent
        self.root = Path.cwd()
        self.images_dir = Path(images_dir)
        self.images_zip = Path(images_zip) if images_zip else None
        self._ensure_images_source()

    def _ensure_images_source(self) -> None:
        if self.images_dir.exists():
            return

        zip_path = self.images_zip or self.root / "images.zip"
        if zip_path.exists():
            self._extract_zip(zip_path, self.images_dir)
            return

        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _safe_extract(self, zip_file: zipfile.ZipFile, destination: Path) -> None:
        destination_resolved = destination.resolve()
        for member in zip_file.infolist():
            member_path = PurePosixPath(member.filename)
            if any(part in {"", ".", ".."} for part in member_path.parts):
                raise ImageResolutionError(f"Unsafe ZIP entry skipped: {member.filename}")
            target = (destination / member_path).resolve()
            if not str(target).startswith(str(destination_resolved)):
                raise ImageResolutionError(f"Unsafe ZIP path detected: {member.filename}")
        zip_file.extractall(destination)

    def _extract_zip(self, zip_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            self._safe_extract(archive, destination)

    def _candidate_paths(self, name: str) -> list[Path]:
        raw = Path(name)
        candidates = [
            raw,
            self.root / raw,
            self.manifest_parent / raw,
            self.images_dir / raw,
        ]
        if raw.is_absolute():
            candidates.insert(0, raw)
        return candidates

    def resolve(self, name: str | None) -> Path | None:
        if not name:
            return None

        for candidate in self._candidate_paths(name):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                return candidate

        basename = Path(name).name
        matches = [
            path
            for path in self.images_dir.rglob(basename)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if matches:
            return matches[0]
        return None

    def validate_images(self, jobs: list[dict]) -> list[str]:
        errors: list[str] = []
        for job in jobs:
            row = job["row"]
            for key in ("start_image", "end_image"):
                value = job.get(key, "")
                if not value:
                    continue
                resolved = self.resolve(value)
                if resolved is None:
                    errors.append(f"Row {row}: {key} image not found: {value}")
                    continue
                try:
                    with Image.open(resolved) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError) as exc:
                    errors.append(f"Row {row}: corrupt or unreadable {key} image {resolved}: {exc}")
                else:
                    job[f"{key}_path"] = str(resolved)
        return errors

    def apply_to_jobs(self, jobs: list[dict]) -> list[str]:
        return self.validate_images(jobs)


def validate_images_and_report(
    jobs: list[dict],
    manifest_path: str | Path,
    images_dir: str | Path,
    images_zip: str | Path | None,
    validation: ValidationResult,
) -> ValidationResult:
    resolver = ImageResolver(manifest_path, images_dir, images_zip)
    validation.errors.extend(resolver.apply_to_jobs(jobs))
    validation.passed = not validation.errors
    if validation.report_path:
        validation.write_report(validation.report_path, Path(manifest_path))
    return validation
