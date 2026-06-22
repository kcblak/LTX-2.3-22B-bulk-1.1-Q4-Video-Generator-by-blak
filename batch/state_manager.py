from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StateManager:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self, manifest_path: str | Path, last_completed_row: int | None = None) -> None:
        current = self.load()
        if last_completed_row is not None:
            current["last_completed_row"] = max(int(current.get("last_completed_row", 0)), last_completed_row)
        current["timestamp"] = datetime.now(timezone.utc).isoformat()
        current["manifest"] = str(manifest_path)

        tmp_fd, tmp_name = tempfile.mkstemp(prefix="state.", suffix=".tmp", dir=self.state_dir)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def mark_completed(self, row: int) -> None:
        self.save("", last_completed_row=row)
