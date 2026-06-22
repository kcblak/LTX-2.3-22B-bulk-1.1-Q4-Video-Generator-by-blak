from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from core.logger import Logger


class FaultTolerance:
    def __init__(self, fs, logger: Logger, max_retries: int = 3):
        self.fs = fs
        self.logger = logger
        self.max_retries = max_retries
        self._log = logger.get_logger("FaultTolerance")

    def safe_run(self, func, job, job_id: str, *args, **kwargs):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(job, job_id, *args, **kwargs)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "oom" in msg or "cuda" in msg:
                    last_err = exc
                    self._handle_oom(job_id)
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                raise
        raise RuntimeError(f"Max retries exceeded for job {job_id}: {last_err}")

    def _handle_oom(self, job_id: str) -> None:
        import torch
        import gc
        self._log.warning("OOM detected for job %s — clearing cache", job_id)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    def cleanup_failed_output(self, job_id: str) -> None:
        for path in [
            self.fs.job_output_path(job_id),
            self.fs.job_frames_dir(job_id),
            self.fs.job_thumbnail_path(job_id),
        ]:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

    def resume(self) -> bool:
        checkpoint = self.fs.checkpoint_path()
        return checkpoint.exists()
