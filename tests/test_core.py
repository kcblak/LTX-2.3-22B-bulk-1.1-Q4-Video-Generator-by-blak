"""Tests for the core notebook system."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, ConfigError
from core.fs_manager import FSManager
from core.logger import Logger


def test_fs_manager_creates_dirs():
    cfg = Config({
        "project_name": "TestProject",
        "mode": "headless",
        "device": "cpu",
        "batch_size": 1,
        "max_retries": 1,
        "parallel_jobs": 1,
        "output_format": "mp4",
        "resolution_default": "720p",
        "logging_level": "info",
    })
    fs = FSManager(cfg)
    assert fs.project_root.exists()
    assert fs.input_images_dir.exists()
    assert fs.output_videos_dir.exists()
    assert fs.logs_dir.exists()


def test_fs_manager_paths():
    cfg = Config({
        "project_name": "MyProject",
        "mode": "headless",
        "device": "cpu",
        "batch_size": 1,
        "max_retries": 1,
        "parallel_jobs": 1,
        "output_format": "mp4",
        "resolution_default": "720p",
        "logging_level": "info",
    })
    fs = FSManager(cfg)
    assert "MyProject" in str(fs.project_root)
    assert fs.jobs_csv_path().name == "jobs.csv"
    assert fs.job_output_path("001").name == "job_001.mp4"


def run_tests():
    test_fs_manager_creates_dirs()
    test_fs_manager_paths()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
