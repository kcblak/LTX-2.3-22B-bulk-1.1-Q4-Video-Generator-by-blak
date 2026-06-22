from __future__ import annotations

import gc
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from core.config import Config
from core.fs_manager import FSManager
from core.job_queue import JobQueue
from core.logger import Logger


class VideoPipeline:
    def __init__(self, config: Config, fs: FSManager, logger: Logger, job_queue: JobQueue):
        self.config = config
        self.fs = fs
        self.logger = logger
        self.job_queue = job_queue
        self._model_loaded = False

    def initialize(self) -> None:
        self._log.info("Initializing video pipeline (mode=%s)", self.config.mode)

    def process_job(self, job: dict) -> dict:
        job_id = str(job["row"])
        job_logger = self.logger.job_logger(job_id)
        job_logger.info("Starting job %s", job_id)
        start_time = _now_iso()
        try:
            self.job_queue.mark_running(job)
            output_file = self._run_generation(job, job_logger)
            self.job_queue.mark_completed(job, output_file)
            job_logger.info("Job %s completed successfully: %s", job_id, output_file)
            return {"status": "completed", "output_file": output_file, "job_id": job_id}
        except Exception as exc:
            self.job_queue.mark_failed(job, str(exc))
            job_logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            torch.cuda.empty_cache()
            gc.collect()
            return {"status": "failed", "error": str(exc), "job_id": job_id}

    def _run_generation(self, job: dict, job_logger) -> str:
        from batch.validator import REQUIRED_COLUMNS
        cfg = self.config
        job_logger.info("Loading model for job %s", job["row"])
        load_model_fn = None
        if cfg.model_name.startswith("ltx"):
            try:
                from ltx_engine import load_ltx_model, Video_Generation, get_resolution, snap_to_ltx_frames
                load_model_fn = load_ltx_model
                generation_fn = Video_Generation
                resolve_fn = get_resolution
                frames_fn = snap_to_ltx_frames
            except ImportError as exc:
                raise RuntimeError(f"LTX engine import failed: {exc}")
        elif cfg.model_name.startswith("svd"):
            try:
                from models.stable_video.svd_adapter import load_svd_model, generate_video
                load_model_fn = load_svd_model
                generation_fn = generate_video
                resolve_fn = lambda r, ar: (1024, 576)
                frames_fn = lambda d, fps=cfg.fps: int(d * fps)
            except ImportError as exc:
                raise RuntimeError(f"SVD adapter import failed: {exc}")
        else:
            try:
                from models.custom_model.custom_adapter import load_model, generate
                load_model_fn = load_model
                generation_fn = generate
                resolve_fn = lambda r, ar: (1024, 576)
                frames_fn = lambda d, fps=cfg.fps: int(d * fps)
            except ImportError as exc:
                raise RuntimeError(f"Custom model adapter import failed: {exc}")

        if not self._model_loaded:
            load_model_fn()
            self._model_loaded = True

        resolution = job.get("resolution") or cfg.resolution_default
        aspect_ratio = job.get("aspect_ratio", "16:9 Landscape")
        width, height = resolve_fn(resolution, aspect_ratio)

        duration_str = job.get("duration", "5 Seconds (121 frames)")
        num_frames = frames_fn(5.0, cfg.fps)
        if duration_str:
            try:
                num_frames = int(duration_str.split("(")[1].split(" frames")[0])
            except Exception:
                pass

        seed_raw = job.get("seed", "").strip()
        seed = -1
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError:
                seed = -1

        guide_scale = float(job.get("guide_scale") or cfg.guide_scale)
        steps = int(job.get("steps") or cfg.sampling_steps)

        output_path = self.fs.job_output_path(job_id, cfg.output_format)

        start_image = job.get("start_image_path")
        end_image = job.get("end_image_path")

        kwargs = dict(
            prompt=job.get("prompt", ""),
            input_image_start=start_image,
            input_image_end=end_image,
            seed=seed,
            duration_dropdown=duration_str or "5 Seconds (121 frames)",
            resolution_dropdown=resolution,
            aspect_ratio_dropdown=aspect_ratio,
            guide_scale=guide_scale,
            num_steps=steps,
            output_path=str(output_path),
        )

        job_logger.info(
            "Generation kwargs: prompt=%s, res=%s, frames=%s, seed=%s, steps=%s",
            kwargs["prompt"][:80],
            (width, height),
            num_frames,
            seed,
            steps,
        )

        if cfg.model_name.startswith("ltx"):
            out_path, status = generation_fn(**kwargs)
        else:
            out_path = generation_fn(**kwargs)
            status = "done"

        if out_path is None or not Path(out_path).exists():
            raise RuntimeError(f"Model returned invalid output path: {out_path}")

        return str(out_path)

    @property
    def _log(self):
        return self.logger.get_logger("VideoPipeline")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
