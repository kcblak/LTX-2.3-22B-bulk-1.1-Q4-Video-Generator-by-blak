from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "relativeCreated", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName", "pathname",
                "filename", "module", "levelname", "levelno", "msecs", "thread",
                "threadName", "process", "processName", "taskName", "message",
            }:
                log_entry[key] = value
        return json.dumps(log_entry, default=str)


class Logger:
    def __init__(self, log_dir: Path, level: str = "info"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.level = getattr(logging, level.upper(), logging.INFO)

        self.system_log_path = self.log_dir / "system.log"
        self._handler = logging.FileHandler(self.system_log_path, encoding="utf-8")
        self._handler.setFormatter(StructuredFormatter())
        self._handler.setLevel(self.level)

        self.root = logging.getLogger("LTX")
        self.root.setLevel(self.level)
        self.root.addHandler(self._handler)

    def get_logger(self, name: str = "LTX") -> logging.Logger:
        return self.root.getChild(name)

    def job_logger(self, job_id: str) -> logging.Logger:
        path = self.log_dir / f"job_{job_id}.log"
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(StructuredFormatter())
        handler.setLevel(self.level)
        logger = logging.getLogger(f"LTX.Job.{job_id}")
        logger.setLevel(self.level)
        logger.addHandler(handler)
        return logger

    def close(self) -> None:
        for handler in list(self.root.handlers):
            handler.close()
            self.root.removeHandler(handler)
