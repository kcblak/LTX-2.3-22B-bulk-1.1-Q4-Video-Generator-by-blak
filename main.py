"""Unified CLI entry point for the LTX Notebook System."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.fs_manager import FSManager
from core.logger import Logger
from core.config import Config, ConfigError


def _build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LTX Notebook System - Headless video generation orchestrator",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Initialize project directory structure")
    init_p.add_argument("--project", required=True, help="Project name")

    run_p = sub.add_parser("run", help="Run full pipeline")
    run_p.add_argument("--headless", action="store_true", help="Run without UI")

    sub.add_parser("resume", help="Resume last interrupted run")
    sub.add_parser("status", help="Show job status")
    sub.add_parser("export-logs", help="Export logs to archive")
    parser.add_argument("--dashboard", action="store_true", help="Start web dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    return parser.parse_args()


def init_project(project_name: str) -> None:
    cfg_data = {
        "project_name": project_name,
        "mode": "headless",
        "device": "cuda",
        "batch_size": 1,
        "max_retries": 3,
        "parallel_jobs": 1,
        "output_format": "mp4",
        "resolution_default": "720p",
        "logging_level": "info",
    }
    config = Config(cfg_data)
    fs = FSManager(config)
    fs.write_project_config()
    print(f"Initialized project: {fs.project_root}")
    for k, v in fs.summary().items():
        print(f"  {k}: {v}")


def _load_config(path: str) -> Config:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for field in ["project_name", "mode", "device", "batch_size", "max_retries",
                   "parallel_jobs", "output_format", "resolution_default", "logging_level"]:
        if field not in data:
            raise ConfigError(f"Missing config field: {field}")
    return Config(data)


def run_pipeline(config_path: str, headless: bool = False) -> None:
    from core.notebook_orchestrator import NotebookOrchestrator
    from core.pipeline import VideoPipeline
    from core.job_queue import JobQueue

    cfg = _load_config(config_path)
    fs = FSManager(cfg)
    logger = Logger(fs.logs_dir, cfg.logging_level)
    queue = JobQueue(fs, logger)

    orchestrator = NotebookOrchestrator(Path("notebooks"), fs, logger)
    context = {"config": cfg, "fs": fs, "log": logger.get_logger("Main")}
    orchestrator.run(context)

    validation = queue.load()
    if not validation.passed:
        logger.get_logger("Main").error("Validation failed: %s", validation.errors)
        sys.exit(1)

    pipeline = VideoPipeline(cfg, fs, logger, queue)
    pipeline.initialize()

    jobs = queue.pending_jobs()
    logger.get_logger("Main").info("Processing %d pending jobs", len(jobs))
    for job in jobs:
        result = pipeline.process_job(job)
        logger.get_logger("Main").info(
            "Job %s result: %s", job["row"], result.get("status")
        )


def main() -> None:
    args = _build_parser()
    if args.command == "init":
        init_project(args.project)
    elif args.command == "run":
        run_pipeline(args.config, headless=args.headless)
    elif args.command == "resume":
        run_pipeline(args.config)
    elif args.command == "status":
        cfg = _load_config(args.config)
        fs = FSManager(cfg)
        status_path = fs.status_csv_path()
        if status_path.exists():
            print(status_path.read_text(encoding="utf-8"))
        else:
            print("No status file found.")
    elif args.command == "export-logs":
        cfg = _load_config(args.config)
        fs = FSManager(cfg)
        import zipfile
        out = Path("logs_export.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for log_file in fs.logs_dir.rglob("*"):
                if log_file.is_file():
                    zf.write(log_file, log_file.relative_to(fs.logs_dir))
        print(f"Exported logs to {out}")
    elif getattr(args, "dashboard", False):
        try:
            from web.dashboard import run_dashboard
            run_dashboard(port=getattr(args, "port", 8000))
        except ImportError as exc:
            print(f"Dashboard dependencies missing: {exc}")
            sys.exit(1)
    else:
        print("Use --help for available commands.")


if __name__ == "__main__":
    main()
