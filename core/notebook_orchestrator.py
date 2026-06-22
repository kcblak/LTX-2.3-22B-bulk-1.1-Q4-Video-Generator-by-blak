from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Config
from core.fs_manager import FSManager
from core.logger import Logger


class NotebookCell:
    def __init__(self, name: str, script_path: Path, fs: FSManager):
        self.name = name
        self.script_path = script_path
        self.fs = fs
        self.done_marker = fs.cell_checkpoint_path(name)

    def already_done(self) -> bool:
        return self.done_marker.exists()

    def mark_done(self) -> None:
        self.done_marker.write_text(
            json.dumps({"completed_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.already_done():
            return context
        spec = importlib.util.spec_from_file_location(f"cell_{self.name}", self.script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load cell script: {self.script_path}")
        module = importlib.util.module_from_spec(spec)
        module.__dict__["fs"] = self.fs
        module.__dict__["config"] = context.get("config")
        module.__dict__["context"] = context
        module.__dict__["log"] = context.get("log")
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if hasattr(module, "run"):
            result = module.run(context)
            if isinstance(result, dict):
                context.update(result)
        self.mark_done()
        return context


class NotebookOrchestrator:
    DEFAULT_CELLS = [
        "cell_01_setup",
        "cell_02_load_data",
        "cell_03_model_inference",
        "cell_04_postprocess",
    ]

    def __init__(self, notebooks_dir: Path, fs: FSManager, logger: Logger):
        self.notebooks_dir = notebooks_dir
        self.fs = fs
        self.logger = logger
        self.cells: list[NotebookCell] = []

    def discover_cells(self) -> list[NotebookCell]:
        cells = []
        for name in self.DEFAULT_CELLS:
            script = self.notebooks_dir / f"{name}.py"
            if script.exists():
                cells.append(NotebookCell(name, script, self.fs))
        scripts = sorted(self.notebooks_dir.glob("cell_*.py"))
        extra = [
            NotebookCell(s.stem, s, self.fs)
            for s in scripts
            if s.stem not in {c.name for c in cells}
        ]
        return sorted(cells + extra, key=lambda c: c.name)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        self.cells = self.discover_cells()
        sys_log = self.logger.get_logger("NotebookOrchestrator")
        sys_log.info("Discovered %d cells", len(self.cells))
        for cell in self.cells:
            cell_log = self.logger.job_logger(f"cell_{cell.name}")
            context["log"] = cell_log
            sys_log.info("Running cell: %s", cell.name)
            if cell.already_done():
                sys_log.info("Cell %s already completed, skipping", cell.name)
                continue
            try:
                context = cell.run(context)
            except Exception as exc:
                cell_log.error("Cell %s failed: %s\n%s", cell.name, exc, traceback.format_exc())
                raise
            sys_log.info("Cell %s completed", cell.name)
        return context

    def reset(self) -> None:
        for marker in self.fs.checkpoints_dir.glob("cell_*.done"):
            marker.unlink()
