from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


STATUS_FIELDS = ["row", "status", "output_file", "start_time", "end_time", "duration", "error"]


class StatusWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[int, dict[str, Any]] = {}

    def load(self) -> dict[int, dict[str, Any]]:
        self.rows = {}
        if not self.path.exists():
            return self.rows
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                try:
                    row = int(raw.get("row", 0))
                except ValueError:
                    continue
                self.rows[row] = {
                    "row": row,
                    "status": raw.get("status", "pending"),
                    "output_file": raw.get("output_file", ""),
                    "start_time": raw.get("start_time", ""),
                    "end_time": raw.get("end_time", ""),
                    "duration": raw.get("duration", ""),
                    "error": raw.get("error", ""),
                }
        return self.rows

    def initialize(self, jobs: list[dict[str, Any]]) -> None:
        existing = self.load()
        for job in jobs:
            row = int(job["row"])
            if row not in existing:
                existing[row] = {
                    "row": row,
                    "status": "pending",
                    "output_file": "",
                    "start_time": "",
                    "end_time": "",
                    "duration": "",
                    "error": "",
                }
        self.rows = existing
        self.flush()

    def update(
        self,
        row: int,
        status: str,
        output_file: str = "",
        start_time: str = "",
        end_time: str = "",
        duration: str = "",
        error: str = "",
    ) -> None:
        if row not in self.rows:
            self.rows[row] = {
                "row": row,
                "status": "pending",
                "output_file": "",
                "start_time": "",
                "end_time": "",
                "duration": "",
                "error": "",
            }
        current = self.rows[row]
        current.update(
            {
                "status": status,
                "output_file": output_file,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "error": error,
            }
        )
        self.flush()

    def flush(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
            writer.writeheader()
            for row in sorted(self.rows):
                writer.writerow(self.rows[row])

    def selected_failed_rows(self) -> set[int]:
        self.load()
        return {row for row, data in self.rows.items() if data.get("status") == "failed"}
