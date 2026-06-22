from __future__ import annotations

import gc
from typing import Optional

import torch

from core.logger import Logger


class Performance:
    def __init__(self, logger: Logger):
        self.logger = logger
        self._log = logger.get_logger("Performance")

    def detect_device(self) -> str:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            sm = torch.cuda.get_device_capability(0)
            self._log.info("CUDA enabled: %s (sm_%d%d)", name, sm[0], sm[1])
            return "cuda"
        self._log.info("CUDA not available, falling back to CPU")
        return "cpu"

    def detect_precision(self) -> str:
        if not torch.cuda.is_available():
            return "fp32"
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return "bf16"
        return "fp16"

    @staticmethod
    def empty_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def log_memory(self, tag: str = "") -> dict:
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        info = {"allocated_mb": round(allocated, 1), "reserved_mb": round(reserved, 1), "tag": tag}
        self._log.info("GPU memory [%s]: allocated=%.1fMB reserved=%.1fMB", tag, allocated, reserved)
        return info
